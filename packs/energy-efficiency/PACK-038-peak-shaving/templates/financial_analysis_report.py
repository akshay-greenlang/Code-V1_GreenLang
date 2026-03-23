# -*- coding: utf-8 -*-
"""
FinancialAnalysisReportTemplate - Financial analysis for PACK-038.

Generates comprehensive financial analysis reports for peak shaving
investments showing NPV, IRR, and payback calculations, incentive
and rebate capture analysis, revenue stacking across demand charges,
arbitrage, and ancillary services, sensitivity analysis across key
variables, and Monte Carlo simulation results with probability
distributions.

Sections:
    1. Financial Overview
    2. NPV / IRR / Payback
    3. Incentive Capture
    4. Revenue Stacking
    5. Sensitivity Analysis
    6. Monte Carlo Results
    7. Recommendations

Output Formats:
    - Markdown (with provenance hash comment)
    - HTML (self-contained with inline CSS)
    - JSON (structured with chart data)

Regulatory References:
    - IRS Section 48 (Investment Tax Credit for storage)
    - EU Taxonomy Regulation (sustainable investment criteria)
    - IFRS 16 / ASC 842 (lease accounting for BESS)

Author: GreenLang Team
Version: 38.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODULE_VERSION = "1.0.0"


class FinancialAnalysisReportTemplate:
    """
    Financial analysis report template for peak shaving investments.

    Renders financial analysis reports showing NPV/IRR/payback,
    incentive capture, revenue stacking, sensitivity analysis, and
    Monte Carlo simulation results across markdown, HTML, and JSON
    formats. All outputs include SHA-256 provenance hashing for
    audit trail integrity.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize FinancialAnalysisReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render financial analysis report as Markdown.

        Args:
            data: Financial analysis engine result data.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_financial_overview(data),
            self._md_npv_irr_payback(data),
            self._md_incentive_capture(data),
            self._md_revenue_stacking(data),
            self._md_sensitivity_analysis(data),
            self._md_monte_carlo(data),
            self._md_recommendations(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._generate_provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render financial analysis report as self-contained HTML.

        Args:
            data: Financial analysis engine result data.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_financial_overview(data),
            self._html_npv_irr_payback(data),
            self._html_incentive_capture(data),
            self._html_revenue_stacking(data),
            self._html_sensitivity_analysis(data),
            self._html_monte_carlo(data),
            self._html_recommendations(data),
        ])
        prov = self._generate_provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Financial Analysis Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render financial analysis report as structured JSON.

        Args:
            data: Financial analysis engine result data.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "financial_analysis_report",
            "version": "38.0.0",
            "generated_at": self.generated_at.isoformat(),
            "financial_overview": self._json_financial_overview(data),
            "npv_irr_payback": self._json_npv_irr_payback(data),
            "incentive_capture": data.get("incentive_capture", []),
            "revenue_stacking": data.get("revenue_stacking", []),
            "sensitivity_analysis": data.get("sensitivity_analysis", []),
            "monte_carlo": self._json_monte_carlo(data),
            "recommendations": data.get("recommendations", []),
            "charts": self._json_charts(data),
        }
        prov = self._generate_provenance(json.dumps(result, default=str))
        result["provenance_hash"] = prov
        return result

    # ------------------------------------------------------------------
    # Markdown section renderers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header with project metadata."""
        facility = data.get("facility_name", "Facility")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Financial Analysis Report\n\n"
            f"**Facility:** {facility}  \n"
            f"**Project:** {data.get('project_name', 'Peak Shaving Investment')}  \n"
            f"**Analysis Period:** {data.get('analysis_period_years', 0)} years  \n"
            f"**Discount Rate:** {self._fmt(data.get('discount_rate_pct', 0))}%  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-038 FinancialAnalysisReportTemplate v38.0.0\n\n---"
        )

    def _md_financial_overview(self, data: Dict[str, Any]) -> str:
        """Render financial overview section."""
        overview = data.get("financial_overview", {})
        return (
            "## 1. Financial Overview\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| Total Investment (CAPEX) | {self._format_currency(overview.get('total_capex', 0))} |\n"
            f"| Annual Operating Cost | {self._format_currency(overview.get('annual_opex', 0))} |\n"
            f"| Total Annual Revenue | {self._format_currency(overview.get('total_annual_revenue', 0))} |\n"
            f"| Net Present Value (NPV) | {self._format_currency(overview.get('npv', 0))} |\n"
            f"| Internal Rate of Return | {self._fmt(overview.get('irr_pct', 0))}% |\n"
            f"| Simple Payback Period | {self._fmt(overview.get('payback_years', 0), 1)} years |\n"
            f"| Discounted Payback | {self._fmt(overview.get('discounted_payback_years', 0), 1)} years |\n"
            f"| Benefit-Cost Ratio | {self._fmt(overview.get('benefit_cost_ratio', 0), 2)} |"
        )

    def _md_npv_irr_payback(self, data: Dict[str, Any]) -> str:
        """Render NPV/IRR/payback cash flow table."""
        cashflows = data.get("npv_irr_payback", {}).get("annual_cashflows", [])
        if not cashflows:
            return "## 2. NPV / IRR / Payback\n\n_No cash flow data available._"
        lines = [
            "## 2. NPV / IRR / Payback\n",
            "| Year | Revenue | OPEX | Net Cash Flow | Cumulative | PV |",
            "|-----:|--------:|-----:|-------------:|----------:|---:|",
        ]
        for cf in cashflows:
            lines.append(
                f"| {cf.get('year', 0)} "
                f"| {self._format_currency(cf.get('revenue', 0))} "
                f"| {self._format_currency(cf.get('opex', 0))} "
                f"| {self._format_currency(cf.get('net_cashflow', 0))} "
                f"| {self._format_currency(cf.get('cumulative', 0))} "
                f"| {self._format_currency(cf.get('present_value', 0))} |"
            )
        return "\n".join(lines)

    def _md_incentive_capture(self, data: Dict[str, Any]) -> str:
        """Render incentive capture section."""
        incentives = data.get("incentive_capture", [])
        if not incentives:
            return "## 3. Incentive Capture\n\n_No incentive data available._"
        lines = [
            "## 3. Incentive Capture\n",
            "| Incentive Program | Type | Amount | Eligibility | Status |",
            "|-------------------|------|-------:|------------|--------|",
        ]
        for inc in incentives:
            lines.append(
                f"| {inc.get('program', '-')} "
                f"| {inc.get('type', '-')} "
                f"| {self._format_currency(inc.get('amount', 0))} "
                f"| {inc.get('eligibility', '-')} "
                f"| {inc.get('status', '-')} |"
            )
        total = sum(i.get("amount", 0) for i in incentives)
        lines.append(f"| **Total Incentives** | | **{self._format_currency(total)}** | | |")
        return "\n".join(lines)

    def _md_revenue_stacking(self, data: Dict[str, Any]) -> str:
        """Render revenue stacking section."""
        streams = data.get("revenue_stacking", [])
        if not streams:
            return "## 4. Revenue Stacking\n\n_No revenue stream data available._"
        lines = [
            "## 4. Revenue Stacking\n",
            "| Revenue Stream | Annual Value | Share (%) | Certainty | Notes |",
            "|---------------|----------:|----------:|----------|-------|",
        ]
        total = sum(s.get("annual_value", 0) for s in streams)
        for stream in streams:
            share = self._pct(stream.get("annual_value", 0), total) if total > 0 else "0.0%"
            lines.append(
                f"| {stream.get('stream', '-')} "
                f"| {self._format_currency(stream.get('annual_value', 0))} "
                f"| {share} "
                f"| {stream.get('certainty', '-')} "
                f"| {stream.get('notes', '-')} |"
            )
        lines.append(f"| **Total Stacked Revenue** | **{self._format_currency(total)}** | 100.0% | | |")
        return "\n".join(lines)

    def _md_sensitivity_analysis(self, data: Dict[str, Any]) -> str:
        """Render sensitivity analysis section."""
        analysis = data.get("sensitivity_analysis", [])
        if not analysis:
            return "## 5. Sensitivity Analysis\n\n_No sensitivity data available._"
        lines = [
            "## 5. Sensitivity Analysis\n",
            "| Variable | -20% NPV | Base NPV | +20% NPV | Elasticity |",
            "|----------|--------:|--------:|--------:|----------:|",
        ]
        for item in analysis:
            lines.append(
                f"| {item.get('variable', '-')} "
                f"| {self._format_currency(item.get('npv_low', 0))} "
                f"| {self._format_currency(item.get('npv_base', 0))} "
                f"| {self._format_currency(item.get('npv_high', 0))} "
                f"| {self._fmt(item.get('elasticity', 0), 2)} |"
            )
        return "\n".join(lines)

    def _md_monte_carlo(self, data: Dict[str, Any]) -> str:
        """Render Monte Carlo simulation results section."""
        mc = data.get("monte_carlo", {})
        if not mc:
            return "## 6. Monte Carlo Results\n\n_No Monte Carlo data available._"
        return (
            "## 6. Monte Carlo Results\n\n"
            f"**Simulations:** {self._fmt(mc.get('simulations', 0), 0)}  \n"
            f"**Confidence Level:** {self._fmt(mc.get('confidence_level_pct', 0))}%\n\n"
            "| Metric | P10 | P50 (Median) | P90 | Mean |\n"
            "|--------|----:|------------:|----:|-----:|\n"
            f"| NPV | {self._format_currency(mc.get('npv_p10', 0))} "
            f"| {self._format_currency(mc.get('npv_p50', 0))} "
            f"| {self._format_currency(mc.get('npv_p90', 0))} "
            f"| {self._format_currency(mc.get('npv_mean', 0))} |\n"
            f"| IRR | {self._fmt(mc.get('irr_p10', 0))}% "
            f"| {self._fmt(mc.get('irr_p50', 0))}% "
            f"| {self._fmt(mc.get('irr_p90', 0))}% "
            f"| {self._fmt(mc.get('irr_mean', 0))}% |\n"
            f"| Payback (yrs) | {self._fmt(mc.get('payback_p10', 0), 1)} "
            f"| {self._fmt(mc.get('payback_p50', 0), 1)} "
            f"| {self._fmt(mc.get('payback_p90', 0), 1)} "
            f"| {self._fmt(mc.get('payback_mean', 0), 1)} |\n\n"
            f"**Probability NPV > 0:** {self._fmt(mc.get('prob_npv_positive_pct', 0))}%  \n"
            f"**Value at Risk (5%):** {self._format_currency(mc.get('var_5pct', 0))}"
        )

    def _md_recommendations(self, data: Dict[str, Any]) -> str:
        """Render recommendations section."""
        recs = data.get("recommendations", [])
        if not recs:
            recs = [
                "Proceed with investment given positive NPV and favorable IRR",
                "Capture available incentives to improve payback by 18 months",
                "Stack demand charge and arbitrage revenue for maximum returns",
                "Monitor sensitivity variables quarterly for early risk detection",
            ]
        lines = ["## 7. Recommendations\n"]
        for i, rec in enumerate(recs, 1):
            lines.append(f"{i}. {rec}")
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render report footer."""
        return "---\n\n*Generated by GreenLang PACK-038 Peak Shaving Pack*"

    # ------------------------------------------------------------------
    # HTML section renderers
    # ------------------------------------------------------------------

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        facility = data.get("facility_name", "Facility")
        return (
            f'<h1>Financial Analysis Report</h1>\n'
            f'<p class="subtitle">Facility: {facility} | '
            f'Project: {data.get("project_name", "Peak Shaving Investment")} | '
            f'Period: {data.get("analysis_period_years", 0)} years</p>'
        )

    def _html_financial_overview(self, data: Dict[str, Any]) -> str:
        """Render HTML financial overview cards."""
        o = data.get("financial_overview", {})
        return (
            '<h2>Financial Overview</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Total CAPEX</span>'
            f'<span class="value">{self._format_currency(o.get("total_capex", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">NPV</span>'
            f'<span class="value">{self._format_currency(o.get("npv", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">IRR</span>'
            f'<span class="value">{self._fmt(o.get("irr_pct", 0))}%</span></div>\n'
            f'  <div class="card"><span class="label">Payback</span>'
            f'<span class="value">{self._fmt(o.get("payback_years", 0), 1)} yrs</span></div>\n'
            f'  <div class="card"><span class="label">BCR</span>'
            f'<span class="value">{self._fmt(o.get("benefit_cost_ratio", 0), 2)}</span></div>\n'
            '</div>'
        )

    def _html_npv_irr_payback(self, data: Dict[str, Any]) -> str:
        """Render HTML cash flow table."""
        cashflows = data.get("npv_irr_payback", {}).get("annual_cashflows", [])
        rows = ""
        for cf in cashflows:
            rows += (
                f'<tr><td>{cf.get("year", 0)}</td>'
                f'<td>{self._format_currency(cf.get("revenue", 0))}</td>'
                f'<td>{self._format_currency(cf.get("opex", 0))}</td>'
                f'<td>{self._format_currency(cf.get("net_cashflow", 0))}</td>'
                f'<td>{self._format_currency(cf.get("cumulative", 0))}</td></tr>\n'
            )
        return (
            '<h2>NPV / IRR / Payback</h2>\n'
            '<table>\n<tr><th>Year</th><th>Revenue</th><th>OPEX</th>'
            f'<th>Net Cash Flow</th><th>Cumulative</th></tr>\n{rows}</table>'
        )

    def _html_incentive_capture(self, data: Dict[str, Any]) -> str:
        """Render HTML incentive capture table."""
        incentives = data.get("incentive_capture", [])
        rows = ""
        for inc in incentives:
            rows += (
                f'<tr><td>{inc.get("program", "-")}</td>'
                f'<td>{inc.get("type", "-")}</td>'
                f'<td>{self._format_currency(inc.get("amount", 0))}</td>'
                f'<td>{inc.get("eligibility", "-")}</td>'
                f'<td>{inc.get("status", "-")}</td></tr>\n'
            )
        return (
            '<h2>Incentive Capture</h2>\n'
            '<table>\n<tr><th>Program</th><th>Type</th><th>Amount</th>'
            f'<th>Eligibility</th><th>Status</th></tr>\n{rows}</table>'
        )

    def _html_revenue_stacking(self, data: Dict[str, Any]) -> str:
        """Render HTML revenue stacking table."""
        streams = data.get("revenue_stacking", [])
        rows = ""
        for stream in streams:
            rows += (
                f'<tr><td>{stream.get("stream", "-")}</td>'
                f'<td>{self._format_currency(stream.get("annual_value", 0))}</td>'
                f'<td>{stream.get("certainty", "-")}</td>'
                f'<td>{stream.get("notes", "-")}</td></tr>\n'
            )
        return (
            '<h2>Revenue Stacking</h2>\n'
            '<table>\n<tr><th>Revenue Stream</th><th>Annual Value</th>'
            f'<th>Certainty</th><th>Notes</th></tr>\n{rows}</table>'
        )

    def _html_sensitivity_analysis(self, data: Dict[str, Any]) -> str:
        """Render HTML sensitivity analysis table."""
        analysis = data.get("sensitivity_analysis", [])
        rows = ""
        for item in analysis:
            rows += (
                f'<tr><td>{item.get("variable", "-")}</td>'
                f'<td>{self._format_currency(item.get("npv_low", 0))}</td>'
                f'<td>{self._format_currency(item.get("npv_base", 0))}</td>'
                f'<td>{self._format_currency(item.get("npv_high", 0))}</td>'
                f'<td>{self._fmt(item.get("elasticity", 0), 2)}</td></tr>\n'
            )
        return (
            '<h2>Sensitivity Analysis</h2>\n'
            '<table>\n<tr><th>Variable</th><th>-20% NPV</th><th>Base NPV</th>'
            f'<th>+20% NPV</th><th>Elasticity</th></tr>\n{rows}</table>'
        )

    def _html_monte_carlo(self, data: Dict[str, Any]) -> str:
        """Render HTML Monte Carlo results."""
        mc = data.get("monte_carlo", {})
        return (
            '<h2>Monte Carlo Results</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">P50 NPV</span>'
            f'<span class="value">{self._format_currency(mc.get("npv_p50", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">P50 IRR</span>'
            f'<span class="value">{self._fmt(mc.get("irr_p50", 0))}%</span></div>\n'
            f'  <div class="card"><span class="label">Prob NPV>0</span>'
            f'<span class="value">{self._fmt(mc.get("prob_npv_positive_pct", 0))}%</span></div>\n'
            f'  <div class="card"><span class="label">VaR (5%)</span>'
            f'<span class="value">{self._format_currency(mc.get("var_5pct", 0))}</span></div>\n'
            '</div>'
        )

    def _html_recommendations(self, data: Dict[str, Any]) -> str:
        """Render HTML recommendations."""
        recs = data.get("recommendations", [
            "Proceed with investment given positive NPV",
            "Stack revenue streams for maximum returns",
        ])
        items = "".join(f'<li>{r}</li>\n' for r in recs)
        return f'<h2>Recommendations</h2>\n<ol>\n{items}</ol>'

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_financial_overview(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON financial overview."""
        o = data.get("financial_overview", {})
        return {
            "total_capex": o.get("total_capex", 0),
            "annual_opex": o.get("annual_opex", 0),
            "total_annual_revenue": o.get("total_annual_revenue", 0),
            "npv": o.get("npv", 0),
            "irr_pct": o.get("irr_pct", 0),
            "payback_years": o.get("payback_years", 0),
            "discounted_payback_years": o.get("discounted_payback_years", 0),
            "benefit_cost_ratio": o.get("benefit_cost_ratio", 0),
        }

    def _json_npv_irr_payback(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON NPV/IRR/payback."""
        nip = data.get("npv_irr_payback", {})
        return {
            "npv": nip.get("npv", 0),
            "irr_pct": nip.get("irr_pct", 0),
            "payback_years": nip.get("payback_years", 0),
            "annual_cashflows": nip.get("annual_cashflows", []),
        }

    def _json_monte_carlo(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON Monte Carlo results."""
        mc = data.get("monte_carlo", {})
        return {
            "simulations": mc.get("simulations", 0),
            "confidence_level_pct": mc.get("confidence_level_pct", 0),
            "npv_p10": mc.get("npv_p10", 0),
            "npv_p50": mc.get("npv_p50", 0),
            "npv_p90": mc.get("npv_p90", 0),
            "npv_mean": mc.get("npv_mean", 0),
            "irr_p10": mc.get("irr_p10", 0),
            "irr_p50": mc.get("irr_p50", 0),
            "irr_p90": mc.get("irr_p90", 0),
            "irr_mean": mc.get("irr_mean", 0),
            "prob_npv_positive_pct": mc.get("prob_npv_positive_pct", 0),
            "var_5pct": mc.get("var_5pct", 0),
        }

    def _json_charts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chart data payloads for visualization."""
        cashflows = data.get("npv_irr_payback", {}).get("annual_cashflows", [])
        streams = data.get("revenue_stacking", [])
        sensitivity = data.get("sensitivity_analysis", [])
        return {
            "cashflow_waterfall": {
                "type": "waterfall",
                "items": [
                    {"label": f"Year {cf.get('year', 0)}", "value": cf.get("net_cashflow", 0)}
                    for cf in cashflows
                ],
            },
            "revenue_pie": {
                "type": "pie",
                "labels": [s.get("stream", "") for s in streams],
                "values": [s.get("annual_value", 0) for s in streams],
            },
            "sensitivity_tornado": {
                "type": "tornado",
                "items": [
                    {
                        "label": s.get("variable", ""),
                        "low": s.get("npv_low", 0),
                        "high": s.get("npv_high", 0),
                    }
                    for s in sensitivity
                ],
            },
            "cumulative_cashflow": {
                "type": "line",
                "labels": [str(cf.get("year", 0)) for cf in cashflows],
                "values": [cf.get("cumulative", 0) for cf in cashflows],
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
            ".label{display:block;font-size:0.85em;color:#6c757d;}"
            ".value{display:block;font-size:1.4em;font-weight:700;color:#198754;}"
            ".subtitle{color:#6c757d;font-size:0.95em;}"
        )

    def _format_currency(self, val: Any) -> str:
        """Format a currency value with comma separators.

        Args:
            val: Numeric value to format.

        Returns:
            Formatted currency string (e.g., 'EUR 1,234.00').
        """
        if isinstance(val, (int, float)):
            return f"EUR {val:,.2f}"
        return str(val)

    def _format_power(self, val: Any) -> str:
        """Format a power value with units.

        Args:
            val: Power value in kW.

        Returns:
            Formatted power string (e.g., '1,234.0 kW').
        """
        if isinstance(val, (int, float)):
            return f"{val:,.1f} kW"
        return str(val)

    def _format_energy(self, val: Any) -> str:
        """Format an energy value with units.

        Args:
            val: Energy value in kWh.

        Returns:
            Formatted energy string (e.g., '1,234.00 kWh').
        """
        if isinstance(val, (int, float)):
            return f"{val:,.2f} kWh"
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

    def _pct(self, part: float, whole: float) -> str:
        """Calculate and format a percentage.

        Args:
            part: Numerator value.
            whole: Denominator value.

        Returns:
            Formatted percentage string.
        """
        if whole == 0:
            return "0.0%"
        return f"{(part / whole) * 100:.1f}%"

    def _generate_provenance(self, content: str) -> str:
        """Compute SHA-256 provenance hash.

        Args:
            content: Content string to hash.

        Returns:
            SHA-256 hex digest.
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
