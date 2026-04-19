# -*- coding: utf-8 -*-
"""
ExecutiveSummaryReportTemplate - C-suite summary for PACK-038.

Generates concise 2-4 page executive summary reports for C-suite
audiences showing key peak shaving metrics, demand charge savings
achieved and projected, BESS return on investment, recommended
actions with priority and timeline, and year-over-year performance
comparison.

Sections:
    1. Key Metrics Dashboard
    2. Demand Charge Savings
    3. BESS ROI Summary
    4. Recommended Actions
    5. Year-over-Year Comparison
    6. Strategic Outlook

Output Formats:
    - Markdown (with provenance hash comment)
    - HTML (self-contained with inline CSS)
    - JSON (structured with chart data)

Regulatory References:
    - CSRD/ESRS E1 (energy disclosure)
    - TCFD recommendations (climate risk reporting)
    - SEC climate disclosure rules (investment impact)

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


class ExecutiveSummaryReportTemplate:
    """
    Executive summary report template for C-suite audiences.

    Renders concise 2-4 page executive summaries showing key peak
    shaving metrics, demand charge savings, BESS ROI, recommended
    actions, and YoY comparison across markdown, HTML, and JSON
    formats. All outputs include SHA-256 provenance hashing for
    audit trail integrity.

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
            self._md_demand_charge_savings(data),
            self._md_bess_roi(data),
            self._md_recommended_actions(data),
            self._md_yoy_comparison(data),
            self._md_strategic_outlook(data),
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
            self._html_demand_charge_savings(data),
            self._html_bess_roi(data),
            self._html_recommended_actions(data),
            self._html_yoy_comparison(data),
            self._html_strategic_outlook(data),
        ])
        prov = self._generate_provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Executive Summary - Peak Shaving</title>\n'
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
            "version": "38.0.0",
            "generated_at": self.generated_at.isoformat(),
            "key_metrics": self._json_key_metrics(data),
            "demand_charge_savings": self._json_demand_charge_savings(data),
            "bess_roi": self._json_bess_roi(data),
            "recommended_actions": data.get("recommended_actions", []),
            "yoy_comparison": data.get("yoy_comparison", []),
            "strategic_outlook": data.get("strategic_outlook", {}),
            "charts": self._json_charts(data),
        }
        prov = self._generate_provenance(json.dumps(result, default=str))
        result["provenance_hash"] = prov
        return result

    # ------------------------------------------------------------------
    # Markdown section renderers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header with facility metadata."""
        facility = data.get("facility_name", "Facility")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Peak Shaving Executive Summary\n\n"
            f"**Facility:** {facility}  \n"
            f"**Reporting Period:** {data.get('reporting_period', '')}  \n"
            f"**Prepared For:** {data.get('prepared_for', 'Senior Management')}  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-038 ExecutiveSummaryReportTemplate v38.0.0\n\n---"
        )

    def _md_key_metrics(self, data: Dict[str, Any]) -> str:
        """Render key metrics dashboard section."""
        metrics = data.get("key_metrics", {})
        return (
            "## 1. Key Metrics Dashboard\n\n"
            "| Metric | Value | vs Target | Status |\n"
            "|--------|-------|-----------|--------|\n"
            f"| Peak Demand Reduction | {self._format_power(metrics.get('peak_reduction_kw', 0))} "
            f"| {self._fmt(metrics.get('peak_reduction_vs_target_pct', 0))}% | {metrics.get('peak_reduction_status', '-')} |\n"
            f"| Demand Charge Savings | {self._format_currency(metrics.get('demand_charge_savings', 0))} "
            f"| {self._fmt(metrics.get('savings_vs_target_pct', 0))}% | {metrics.get('savings_status', '-')} |\n"
            f"| BESS Utilization | {self._fmt(metrics.get('bess_utilization_pct', 0))}% "
            f"| {self._fmt(metrics.get('utilization_vs_target_pct', 0))}% | {metrics.get('utilization_status', '-')} |\n"
            f"| Load Factor Improvement | {self._fmt(metrics.get('load_factor_improvement_pct', 0))}% "
            f"| {self._fmt(metrics.get('lf_vs_target_pct', 0))}% | {metrics.get('lf_status', '-')} |\n"
            f"| CP Events Avoided | {metrics.get('cp_events_avoided', 0)}/{metrics.get('cp_events_total', 0)} "
            f"| {self._fmt(metrics.get('cp_vs_target_pct', 0))}% | {metrics.get('cp_status', '-')} |\n"
            f"| Total Annual Savings | {self._format_currency(metrics.get('total_annual_savings', 0))} "
            f"| {self._fmt(metrics.get('total_vs_target_pct', 0))}% | {metrics.get('total_status', '-')} |"
        )

    def _md_demand_charge_savings(self, data: Dict[str, Any]) -> str:
        """Render demand charge savings section."""
        savings = data.get("demand_charge_savings", {})
        monthly = savings.get("monthly_breakdown", [])
        lines = [
            "## 2. Demand Charge Savings\n",
            f"**Total Savings (YTD):** {self._format_currency(savings.get('ytd_savings', 0))}  \n"
            f"**Projected Annual:** {self._format_currency(savings.get('projected_annual', 0))}  \n"
            f"**Budget Target:** {self._format_currency(savings.get('budget_target', 0))}  \n"
            f"**Variance:** {self._format_currency(savings.get('variance', 0))}\n",
        ]
        if monthly:
            lines.extend([
                "| Month | Before (kW) | After (kW) | Reduction | Savings |",
                "|-------|----------:|--------:|----------:|--------:|",
            ])
            for m in monthly:
                lines.append(
                    f"| {m.get('month', '-')} "
                    f"| {self._fmt(m.get('peak_before_kw', 0), 0)} "
                    f"| {self._fmt(m.get('peak_after_kw', 0), 0)} "
                    f"| {self._fmt(m.get('reduction_kw', 0), 0)} kW "
                    f"| {self._format_currency(m.get('savings', 0))} |"
                )
        return "\n".join(lines)

    def _md_bess_roi(self, data: Dict[str, Any]) -> str:
        """Render BESS ROI summary section."""
        roi = data.get("bess_roi", {})
        if not roi:
            return "## 3. BESS ROI Summary\n\n_No BESS ROI data available._"
        return (
            "## 3. BESS ROI Summary\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| Total Investment | {self._format_currency(roi.get('total_investment', 0))} |\n"
            f"| Annual Revenue (All Streams) | {self._format_currency(roi.get('annual_revenue', 0))} |\n"
            f"| Net Present Value | {self._format_currency(roi.get('npv', 0))} |\n"
            f"| Internal Rate of Return | {self._fmt(roi.get('irr_pct', 0))}% |\n"
            f"| Payback Period | {self._fmt(roi.get('payback_years', 0), 1)} years |\n"
            f"| Battery Health (SOH) | {self._fmt(roi.get('soh_pct', 0))}% |\n"
            f"| Lifetime Revenue Captured | {self._format_currency(roi.get('lifetime_revenue', 0))} |\n"
            f"| ROI to Date | {self._fmt(roi.get('roi_to_date_pct', 0))}% |"
        )

    def _md_recommended_actions(self, data: Dict[str, Any]) -> str:
        """Render recommended actions section."""
        actions = data.get("recommended_actions", [])
        if not actions:
            actions = [
                {"action": "Expand BESS capacity by 500 kW for additional peak shaving",
                 "priority": "High", "timeline": "Q2 2026", "impact": "EUR 120,000/yr"},
                {"action": "Implement automated CP prediction and response system",
                 "priority": "High", "timeline": "Q1 2026", "impact": "EUR 85,000/yr"},
                {"action": "Deploy load shifting for HVAC and process loads",
                 "priority": "Medium", "timeline": "Q3 2026", "impact": "EUR 45,000/yr"},
                {"action": "Negotiate rate schedule change with utility",
                 "priority": "Medium", "timeline": "Q4 2026", "impact": "EUR 30,000/yr"},
            ]
        lines = [
            "## 4. Recommended Actions\n",
            "| # | Action | Priority | Timeline | Est. Impact |",
            "|---|--------|----------|----------|-------------|",
        ]
        for i, action in enumerate(actions, 1):
            if isinstance(action, dict):
                lines.append(
                    f"| {i} | {action.get('action', '-')} "
                    f"| {action.get('priority', '-')} "
                    f"| {action.get('timeline', '-')} "
                    f"| {action.get('impact', '-')} |"
                )
            else:
                lines.append(f"| {i} | {action} | - | - | - |")
        return "\n".join(lines)

    def _md_yoy_comparison(self, data: Dict[str, Any]) -> str:
        """Render year-over-year comparison section."""
        yoy = data.get("yoy_comparison", [])
        if not yoy:
            return "## 5. Year-over-Year Comparison\n\n_No YoY data available._"
        lines = [
            "## 5. Year-over-Year Comparison\n",
            "| Metric | Prior Year | Current Year | Change | Trend |",
            "|--------|----------:|----------:|------:|-------|",
        ]
        for item in yoy:
            change = item.get("change_pct", 0)
            trend = "Up" if change > 0 else ("Down" if change < 0 else "Flat")
            lines.append(
                f"| {item.get('metric', '-')} "
                f"| {item.get('prior_year', '-')} "
                f"| {item.get('current_year', '-')} "
                f"| {self._fmt(change)}% "
                f"| {trend} |"
            )
        return "\n".join(lines)

    def _md_strategic_outlook(self, data: Dict[str, Any]) -> str:
        """Render strategic outlook section."""
        outlook = data.get("strategic_outlook", {})
        if not outlook:
            return "## 6. Strategic Outlook\n\n_No strategic outlook data available._"
        points = outlook.get("key_points", [])
        lines = ["## 6. Strategic Outlook\n"]
        for point in points:
            lines.append(f"- {point}")
        if outlook.get("three_year_savings"):
            lines.append(
                f"\n**Three-Year Projected Savings:** "
                f"{self._format_currency(outlook.get('three_year_savings', 0))}"
            )
        if outlook.get("risk_factors"):
            lines.append("\n**Key Risk Factors:**")
            for risk in outlook.get("risk_factors", []):
                lines.append(f"- {risk}")
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
            f'<h1>Peak Shaving Executive Summary</h1>\n'
            f'<p class="subtitle">Facility: {facility} | '
            f'Period: {data.get("reporting_period", "-")} | '
            f'Prepared For: {data.get("prepared_for", "Senior Management")}</p>'
        )

    def _html_key_metrics(self, data: Dict[str, Any]) -> str:
        """Render HTML key metrics dashboard cards."""
        m = data.get("key_metrics", {})
        return (
            '<h2>Key Metrics Dashboard</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Peak Reduction</span>'
            f'<span class="value">{self._fmt(m.get("peak_reduction_kw", 0), 0)} kW</span></div>\n'
            f'  <div class="card"><span class="label">Demand Savings</span>'
            f'<span class="value">{self._format_currency(m.get("demand_charge_savings", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">BESS Utilization</span>'
            f'<span class="value">{self._fmt(m.get("bess_utilization_pct", 0))}%</span></div>\n'
            f'  <div class="card"><span class="label">CP Events Avoided</span>'
            f'<span class="value">{m.get("cp_events_avoided", 0)}/{m.get("cp_events_total", 0)}</span></div>\n'
            f'  <div class="card"><span class="label">Total Annual Savings</span>'
            f'<span class="value">{self._format_currency(m.get("total_annual_savings", 0))}</span></div>\n'
            '</div>'
        )

    def _html_demand_charge_savings(self, data: Dict[str, Any]) -> str:
        """Render HTML demand charge savings table."""
        savings = data.get("demand_charge_savings", {})
        monthly = savings.get("monthly_breakdown", [])
        rows = ""
        for m in monthly:
            rows += (
                f'<tr><td>{m.get("month", "-")}</td>'
                f'<td>{self._fmt(m.get("peak_before_kw", 0), 0)}</td>'
                f'<td>{self._fmt(m.get("peak_after_kw", 0), 0)}</td>'
                f'<td>{self._fmt(m.get("reduction_kw", 0), 0)}</td>'
                f'<td>{self._format_currency(m.get("savings", 0))}</td></tr>\n'
            )
        return (
            '<h2>Demand Charge Savings</h2>\n'
            '<table>\n<tr><th>Month</th><th>Before (kW)</th><th>After (kW)</th>'
            f'<th>Reduction</th><th>Savings</th></tr>\n{rows}</table>'
        )

    def _html_bess_roi(self, data: Dict[str, Any]) -> str:
        """Render HTML BESS ROI summary."""
        r = data.get("bess_roi", {})
        return (
            '<h2>BESS ROI Summary</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Investment</span>'
            f'<span class="value">{self._format_currency(r.get("total_investment", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">NPV</span>'
            f'<span class="value">{self._format_currency(r.get("npv", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">IRR</span>'
            f'<span class="value">{self._fmt(r.get("irr_pct", 0))}%</span></div>\n'
            f'  <div class="card"><span class="label">Payback</span>'
            f'<span class="value">{self._fmt(r.get("payback_years", 0), 1)} yrs</span></div>\n'
            f'  <div class="card"><span class="label">SOH</span>'
            f'<span class="value">{self._fmt(r.get("soh_pct", 0))}%</span></div>\n'
            '</div>'
        )

    def _html_recommended_actions(self, data: Dict[str, Any]) -> str:
        """Render HTML recommended actions table."""
        actions = data.get("recommended_actions", [])
        rows = ""
        for i, action in enumerate(actions, 1):
            if isinstance(action, dict):
                prio = action.get("priority", "").lower()
                rows += (
                    f'<tr><td>{i}</td>'
                    f'<td>{action.get("action", "-")}</td>'
                    f'<td class="severity-{prio}">{action.get("priority", "-")}</td>'
                    f'<td>{action.get("timeline", "-")}</td>'
                    f'<td>{action.get("impact", "-")}</td></tr>\n'
                )
            else:
                rows += f'<tr><td>{i}</td><td>{action}</td><td>-</td><td>-</td><td>-</td></tr>\n'
        return (
            '<h2>Recommended Actions</h2>\n'
            '<table>\n<tr><th>#</th><th>Action</th><th>Priority</th>'
            f'<th>Timeline</th><th>Impact</th></tr>\n{rows}</table>'
        )

    def _html_yoy_comparison(self, data: Dict[str, Any]) -> str:
        """Render HTML year-over-year comparison table."""
        yoy = data.get("yoy_comparison", [])
        rows = ""
        for item in yoy:
            change = item.get("change_pct", 0)
            color = "#198754" if change > 0 else ("#dc3545" if change < 0 else "#6c757d")
            rows += (
                f'<tr><td>{item.get("metric", "-")}</td>'
                f'<td>{item.get("prior_year", "-")}</td>'
                f'<td>{item.get("current_year", "-")}</td>'
                f'<td style="color:{color};font-weight:700">{self._fmt(change)}%</td></tr>\n'
            )
        return (
            '<h2>Year-over-Year Comparison</h2>\n'
            '<table>\n<tr><th>Metric</th><th>Prior Year</th><th>Current Year</th>'
            f'<th>Change</th></tr>\n{rows}</table>'
        )

    def _html_strategic_outlook(self, data: Dict[str, Any]) -> str:
        """Render HTML strategic outlook."""
        outlook = data.get("strategic_outlook", {})
        points = outlook.get("key_points", [])
        items = "".join(f'<li>{p}</li>\n' for p in points)
        return (
            '<h2>Strategic Outlook</h2>\n'
            f'<ul>\n{items}</ul>\n'
            f'<p><strong>Three-Year Projected Savings:</strong> '
            f'{self._format_currency(outlook.get("three_year_savings", 0))}</p>'
        )

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_key_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON key metrics."""
        m = data.get("key_metrics", {})
        return {
            "peak_reduction_kw": m.get("peak_reduction_kw", 0),
            "demand_charge_savings": m.get("demand_charge_savings", 0),
            "bess_utilization_pct": m.get("bess_utilization_pct", 0),
            "load_factor_improvement_pct": m.get("load_factor_improvement_pct", 0),
            "cp_events_avoided": m.get("cp_events_avoided", 0),
            "cp_events_total": m.get("cp_events_total", 0),
            "total_annual_savings": m.get("total_annual_savings", 0),
        }

    def _json_demand_charge_savings(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON demand charge savings."""
        s = data.get("demand_charge_savings", {})
        return {
            "ytd_savings": s.get("ytd_savings", 0),
            "projected_annual": s.get("projected_annual", 0),
            "budget_target": s.get("budget_target", 0),
            "variance": s.get("variance", 0),
            "monthly_breakdown": s.get("monthly_breakdown", []),
        }

    def _json_bess_roi(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON BESS ROI."""
        r = data.get("bess_roi", {})
        return {
            "total_investment": r.get("total_investment", 0),
            "annual_revenue": r.get("annual_revenue", 0),
            "npv": r.get("npv", 0),
            "irr_pct": r.get("irr_pct", 0),
            "payback_years": r.get("payback_years", 0),
            "soh_pct": r.get("soh_pct", 0),
            "lifetime_revenue": r.get("lifetime_revenue", 0),
            "roi_to_date_pct": r.get("roi_to_date_pct", 0),
        }

    def _json_charts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chart data payloads for visualization."""
        monthly = data.get("demand_charge_savings", {}).get("monthly_breakdown", [])
        yoy = data.get("yoy_comparison", [])
        return {
            "savings_trend": {
                "type": "bar",
                "labels": [m.get("month", "") for m in monthly],
                "values": [m.get("savings", 0) for m in monthly],
            },
            "peak_reduction": {
                "type": "grouped_bar",
                "labels": [m.get("month", "") for m in monthly],
                "series": {
                    "before": [m.get("peak_before_kw", 0) for m in monthly],
                    "after": [m.get("peak_after_kw", 0) for m in monthly],
                },
            },
            "yoy_comparison": {
                "type": "grouped_bar",
                "labels": [y.get("metric", "") for y in yoy],
                "series": {
                    "prior": [y.get("prior_year_value", 0) for y in yoy],
                    "current": [y.get("current_year_value", 0) for y in yoy],
                },
            },
            "roi_gauge": {
                "type": "gauge",
                "value": data.get("bess_roi", {}).get("roi_to_date_pct", 0),
                "max": 200,
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
            ".severity-high{color:#dc3545;font-weight:700;}"
            ".severity-medium{color:#fd7e14;font-weight:600;}"
            ".severity-low{color:#198754;font-weight:500;}"
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
