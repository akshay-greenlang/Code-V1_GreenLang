# -*- coding: utf-8 -*-
"""
DemandChargeReportTemplate - Demand charge decomposition for PACK-038.

Generates comprehensive demand charge analysis reports showing charge
component breakdown (facility, transmission, distribution), marginal
demand values, tariff structure comparison across rate options,
projected charges under various peak reduction scenarios.

Sections:
    1. Charge Summary
    2. Component Breakdown
    3. Marginal Demand Values
    4. Tariff Comparison
    5. Projected Charges
    6. Rate Optimization
    7. Recommendations

Output Formats:
    - Markdown (with provenance hash comment)
    - HTML (self-contained with inline CSS)
    - JSON (structured with chart data)

Regulatory References:
    - PURPA Section 111(d) (demand charge reform)
    - EU Electricity Directive 2019/944 (network tariff structure)
    - FERC Form 1 (utility rate schedules)

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


class DemandChargeReportTemplate:
    """
    Demand charge decomposition report template.

    Renders demand charge analysis reports showing component breakdown,
    marginal demand values, tariff comparison across rate schedules,
    and projected charges under peak reduction scenarios across
    markdown, HTML, and JSON formats. All outputs include SHA-256
    provenance hashing for audit trail integrity.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize DemandChargeReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render demand charge report as Markdown.

        Args:
            data: Demand charge engine result data.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_charge_summary(data),
            self._md_component_breakdown(data),
            self._md_marginal_values(data),
            self._md_tariff_comparison(data),
            self._md_projected_charges(data),
            self._md_rate_optimization(data),
            self._md_recommendations(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._generate_provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render demand charge report as self-contained HTML.

        Args:
            data: Demand charge engine result data.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_charge_summary(data),
            self._html_component_breakdown(data),
            self._html_marginal_values(data),
            self._html_tariff_comparison(data),
            self._html_projected_charges(data),
            self._html_rate_optimization(data),
            self._html_recommendations(data),
        ])
        prov = self._generate_provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Demand Charge Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render demand charge report as structured JSON.

        Args:
            data: Demand charge engine result data.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "demand_charge_report",
            "version": "38.0.0",
            "generated_at": self.generated_at.isoformat(),
            "charge_summary": self._json_charge_summary(data),
            "component_breakdown": data.get("component_breakdown", []),
            "marginal_values": data.get("marginal_values", []),
            "tariff_comparison": data.get("tariff_comparison", []),
            "projected_charges": data.get("projected_charges", []),
            "rate_optimization": data.get("rate_optimization", {}),
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
        """Render markdown header with facility metadata."""
        facility = data.get("facility_name", "Facility")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Demand Charge Analysis Report\n\n"
            f"**Facility:** {facility}  \n"
            f"**Billing Period:** {data.get('billing_period', '')}  \n"
            f"**Rate Schedule:** {data.get('rate_schedule', '')}  \n"
            f"**Peak Demand:** {self._format_power(data.get('peak_demand_kw', 0))}  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-038 DemandChargeReportTemplate v38.0.0\n\n---"
        )

    def _md_charge_summary(self, data: Dict[str, Any]) -> str:
        """Render charge summary section."""
        summary = data.get("charge_summary", {})
        return (
            "## 1. Charge Summary\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| Total Demand Charges | {self._format_currency(summary.get('total_demand_charges', 0))} |\n"
            f"| Facility Demand Charge | {self._format_currency(summary.get('facility_charge', 0))} |\n"
            f"| Transmission Demand Charge | {self._format_currency(summary.get('transmission_charge', 0))} |\n"
            f"| Distribution Demand Charge | {self._format_currency(summary.get('distribution_charge', 0))} |\n"
            f"| Billed Demand (kW) | {self._format_power(summary.get('billed_demand_kw', 0))} |\n"
            f"| Demand as % of Total Bill | {self._fmt(summary.get('demand_pct_of_bill', 0))}% |\n"
            f"| Cost per kW | {self._format_currency(summary.get('cost_per_kw', 0))}/kW |\n"
            f"| Ratchet Impact | {self._format_currency(summary.get('ratchet_impact', 0))} |"
        )

    def _md_component_breakdown(self, data: Dict[str, Any]) -> str:
        """Render demand charge component breakdown section."""
        components = data.get("component_breakdown", [])
        if not components:
            return "## 2. Component Breakdown\n\n_No component data available._"
        lines = [
            "## 2. Component Breakdown\n",
            "| Component | Rate (EUR/kW) | Billed kW | Charge (EUR) | Share (%) |",
            "|-----------|------------:|--------:|------------:|----------:|",
        ]
        for comp in components:
            lines.append(
                f"| {comp.get('component', '-')} "
                f"| {self._fmt(comp.get('rate_per_kw', 0))} "
                f"| {self._fmt(comp.get('billed_kw', 0), 1)} "
                f"| {self._fmt(comp.get('charge', 0))} "
                f"| {self._fmt(comp.get('share_pct', 0))}% |"
            )
        return "\n".join(lines)

    def _md_marginal_values(self, data: Dict[str, Any]) -> str:
        """Render marginal demand values section."""
        marginals = data.get("marginal_values", [])
        if not marginals:
            return "## 3. Marginal Demand Values\n\n_No marginal value data available._"
        lines = [
            "## 3. Marginal Demand Values\n",
            "| kW Reduction | Monthly Savings | Annual Savings | Marginal Value |",
            "|----------:|-------------:|-------------:|-------------:|",
        ]
        for m in marginals:
            lines.append(
                f"| {self._fmt(m.get('kw_reduction', 0), 0)} kW "
                f"| {self._format_currency(m.get('monthly_savings', 0))} "
                f"| {self._format_currency(m.get('annual_savings', 0))} "
                f"| {self._format_currency(m.get('marginal_value', 0))}/kW |"
            )
        return "\n".join(lines)

    def _md_tariff_comparison(self, data: Dict[str, Any]) -> str:
        """Render tariff comparison section."""
        tariffs = data.get("tariff_comparison", [])
        if not tariffs:
            return "## 4. Tariff Comparison\n\n_No tariff comparison data available._"
        lines = [
            "## 4. Tariff Comparison\n",
            "| Rate Schedule | Demand Rate | Monthly Cost | Annual Cost | vs Current |",
            "|--------------|----------:|----------:|----------:|----------:|",
        ]
        for t in tariffs:
            diff = t.get("vs_current_pct", 0)
            marker = "+" if diff > 0 else ""
            lines.append(
                f"| {t.get('rate_schedule', '-')} "
                f"| {self._format_currency(t.get('demand_rate', 0))}/kW "
                f"| {self._format_currency(t.get('monthly_cost', 0))} "
                f"| {self._format_currency(t.get('annual_cost', 0))} "
                f"| {marker}{self._fmt(diff)}% |"
            )
        return "\n".join(lines)

    def _md_projected_charges(self, data: Dict[str, Any]) -> str:
        """Render projected charges under peak reduction scenarios."""
        projections = data.get("projected_charges", [])
        if not projections:
            return "## 5. Projected Charges\n\n_No projection data available._"
        lines = [
            "## 5. Projected Charges\n",
            "| Scenario | Peak kW | Monthly Charge | Annual Charge | Savings |",
            "|----------|------:|-------------:|-------------:|--------:|",
        ]
        for proj in projections:
            lines.append(
                f"| {proj.get('scenario', '-')} "
                f"| {self._fmt(proj.get('peak_kw', 0), 0)} "
                f"| {self._format_currency(proj.get('monthly_charge', 0))} "
                f"| {self._format_currency(proj.get('annual_charge', 0))} "
                f"| {self._format_currency(proj.get('savings', 0))} |"
            )
        return "\n".join(lines)

    def _md_rate_optimization(self, data: Dict[str, Any]) -> str:
        """Render rate optimization analysis section."""
        optimization = data.get("rate_optimization", {})
        if not optimization:
            return "## 6. Rate Optimization\n\n_No rate optimization data available._"
        return (
            "## 6. Rate Optimization\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| Current Rate Schedule | {optimization.get('current_schedule', '-')} |\n"
            f"| Recommended Schedule | {optimization.get('recommended_schedule', '-')} |\n"
            f"| Current Annual Cost | {self._format_currency(optimization.get('current_annual_cost', 0))} |\n"
            f"| Optimized Annual Cost | {self._format_currency(optimization.get('optimized_annual_cost', 0))} |\n"
            f"| Potential Savings | {self._format_currency(optimization.get('potential_savings', 0))} |\n"
            f"| Switching Cost | {self._format_currency(optimization.get('switching_cost', 0))} |\n"
            f"| Payback Period | {optimization.get('payback_months', 0)} months |"
        )

    def _md_recommendations(self, data: Dict[str, Any]) -> str:
        """Render recommendations section."""
        recs = data.get("recommendations", [])
        if not recs:
            recs = [
                "Evaluate rate schedule switch to reduce demand charges by 15%",
                "Target peak reduction of top 5 billing peaks for maximum savings",
                "Install BESS to clip demand spikes exceeding ratchet threshold",
                "Implement demand monitoring with 15-minute alert thresholds",
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
            f'<h1>Demand Charge Analysis Report</h1>\n'
            f'<p class="subtitle">Facility: {facility} | '
            f'Rate Schedule: {data.get("rate_schedule", "-")} | '
            f'Billing Period: {data.get("billing_period", "-")}</p>'
        )

    def _html_charge_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML charge summary cards."""
        s = data.get("charge_summary", {})
        return (
            '<h2>Charge Summary</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Total Demand Charges</span>'
            f'<span class="value">{self._format_currency(s.get("total_demand_charges", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Billed Demand</span>'
            f'<span class="value">{self._fmt(s.get("billed_demand_kw", 0), 0)} kW</span></div>\n'
            f'  <div class="card"><span class="label">Cost per kW</span>'
            f'<span class="value">{self._format_currency(s.get("cost_per_kw", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Demand % of Bill</span>'
            f'<span class="value">{self._fmt(s.get("demand_pct_of_bill", 0))}%</span></div>\n'
            f'  <div class="card"><span class="label">Ratchet Impact</span>'
            f'<span class="value">{self._format_currency(s.get("ratchet_impact", 0))}</span></div>\n'
            '</div>'
        )

    def _html_component_breakdown(self, data: Dict[str, Any]) -> str:
        """Render HTML component breakdown table."""
        components = data.get("component_breakdown", [])
        rows = ""
        for comp in components:
            rows += (
                f'<tr><td>{comp.get("component", "-")}</td>'
                f'<td>{self._fmt(comp.get("rate_per_kw", 0))}</td>'
                f'<td>{self._fmt(comp.get("billed_kw", 0), 1)}</td>'
                f'<td>{self._fmt(comp.get("charge", 0))}</td>'
                f'<td>{self._fmt(comp.get("share_pct", 0))}%</td></tr>\n'
            )
        return (
            '<h2>Component Breakdown</h2>\n'
            '<table>\n<tr><th>Component</th><th>Rate (EUR/kW)</th>'
            f'<th>Billed kW</th><th>Charge (EUR)</th><th>Share</th></tr>\n{rows}</table>'
        )

    def _html_marginal_values(self, data: Dict[str, Any]) -> str:
        """Render HTML marginal values table."""
        marginals = data.get("marginal_values", [])
        rows = ""
        for m in marginals:
            rows += (
                f'<tr><td>{self._fmt(m.get("kw_reduction", 0), 0)} kW</td>'
                f'<td>{self._format_currency(m.get("monthly_savings", 0))}</td>'
                f'<td>{self._format_currency(m.get("annual_savings", 0))}</td>'
                f'<td>{self._format_currency(m.get("marginal_value", 0))}/kW</td></tr>\n'
            )
        return (
            '<h2>Marginal Demand Values</h2>\n'
            '<table>\n<tr><th>kW Reduction</th><th>Monthly Savings</th>'
            f'<th>Annual Savings</th><th>Marginal Value</th></tr>\n{rows}</table>'
        )

    def _html_tariff_comparison(self, data: Dict[str, Any]) -> str:
        """Render HTML tariff comparison table."""
        tariffs = data.get("tariff_comparison", [])
        rows = ""
        for t in tariffs:
            diff = t.get("vs_current_pct", 0)
            color = "#dc3545" if diff > 0 else "#198754"
            marker = "+" if diff > 0 else ""
            rows += (
                f'<tr><td>{t.get("rate_schedule", "-")}</td>'
                f'<td>{self._format_currency(t.get("demand_rate", 0))}/kW</td>'
                f'<td>{self._format_currency(t.get("monthly_cost", 0))}</td>'
                f'<td>{self._format_currency(t.get("annual_cost", 0))}</td>'
                f'<td style="color:{color}">{marker}{self._fmt(diff)}%</td></tr>\n'
            )
        return (
            '<h2>Tariff Comparison</h2>\n'
            '<table>\n<tr><th>Rate Schedule</th><th>Demand Rate</th>'
            f'<th>Monthly Cost</th><th>Annual Cost</th><th>vs Current</th></tr>\n{rows}</table>'
        )

    def _html_projected_charges(self, data: Dict[str, Any]) -> str:
        """Render HTML projected charges table."""
        projections = data.get("projected_charges", [])
        rows = ""
        for proj in projections:
            rows += (
                f'<tr><td>{proj.get("scenario", "-")}</td>'
                f'<td>{self._fmt(proj.get("peak_kw", 0), 0)}</td>'
                f'<td>{self._format_currency(proj.get("monthly_charge", 0))}</td>'
                f'<td>{self._format_currency(proj.get("annual_charge", 0))}</td>'
                f'<td>{self._format_currency(proj.get("savings", 0))}</td></tr>\n'
            )
        return (
            '<h2>Projected Charges</h2>\n'
            '<table>\n<tr><th>Scenario</th><th>Peak kW</th>'
            f'<th>Monthly</th><th>Annual</th><th>Savings</th></tr>\n{rows}</table>'
        )

    def _html_rate_optimization(self, data: Dict[str, Any]) -> str:
        """Render HTML rate optimization summary."""
        o = data.get("rate_optimization", {})
        return (
            '<h2>Rate Optimization</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Current Schedule</span>'
            f'<span class="value">{o.get("current_schedule", "-")}</span></div>\n'
            f'  <div class="card"><span class="label">Recommended</span>'
            f'<span class="value">{o.get("recommended_schedule", "-")}</span></div>\n'
            f'  <div class="card"><span class="label">Potential Savings</span>'
            f'<span class="value">{self._format_currency(o.get("potential_savings", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Payback</span>'
            f'<span class="value">{o.get("payback_months", 0)} months</span></div>\n'
            '</div>'
        )

    def _html_recommendations(self, data: Dict[str, Any]) -> str:
        """Render HTML recommendations."""
        recs = data.get("recommendations", [
            "Evaluate rate schedule switch for demand charge reduction",
            "Target peak reduction for maximum marginal savings",
        ])
        items = "".join(f'<li>{r}</li>\n' for r in recs)
        return f'<h2>Recommendations</h2>\n<ol>\n{items}</ol>'

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_charge_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON charge summary."""
        s = data.get("charge_summary", {})
        return {
            "total_demand_charges": s.get("total_demand_charges", 0),
            "facility_charge": s.get("facility_charge", 0),
            "transmission_charge": s.get("transmission_charge", 0),
            "distribution_charge": s.get("distribution_charge", 0),
            "billed_demand_kw": s.get("billed_demand_kw", 0),
            "demand_pct_of_bill": s.get("demand_pct_of_bill", 0),
            "cost_per_kw": s.get("cost_per_kw", 0),
            "ratchet_impact": s.get("ratchet_impact", 0),
        }

    def _json_charts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chart data payloads for visualization."""
        components = data.get("component_breakdown", [])
        marginals = data.get("marginal_values", [])
        tariffs = data.get("tariff_comparison", [])
        projections = data.get("projected_charges", [])
        return {
            "component_pie": {
                "type": "pie",
                "labels": [c.get("component", "") for c in components],
                "values": [c.get("charge", 0) for c in components],
            },
            "marginal_value_line": {
                "type": "line",
                "labels": [str(m.get("kw_reduction", 0)) for m in marginals],
                "values": [m.get("marginal_value", 0) for m in marginals],
            },
            "tariff_comparison_bar": {
                "type": "grouped_bar",
                "labels": [t.get("rate_schedule", "") for t in tariffs],
                "values": [t.get("annual_cost", 0) for t in tariffs],
            },
            "projection_waterfall": {
                "type": "waterfall",
                "items": [
                    {"label": p.get("scenario", ""), "value": p.get("savings", 0)}
                    for p in projections
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
            val: Energy value in MWh.

        Returns:
            Formatted energy string (e.g., '1,234.00 MWh').
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
