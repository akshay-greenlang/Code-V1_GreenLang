# -*- coding: utf-8 -*-
"""
BESSSizingReportTemplate - BESS optimization for PACK-038.

Generates comprehensive Battery Energy Storage System (BESS) sizing
and optimization reports showing technology comparison across lithium-ion,
flow, and sodium-ion chemistries, dispatch simulation results for peak
shaving scenarios, degradation projections over system lifetime, and
detailed financial analysis including NPV, IRR, and payback metrics.

Sections:
    1. BESS Overview
    2. Technology Comparison
    3. Dispatch Simulation Results
    4. Degradation Projections
    5. Financial Analysis
    6. Sensitivity Analysis
    7. Recommendations

Output Formats:
    - Markdown (with provenance hash comment)
    - HTML (self-contained with inline CSS)
    - JSON (structured with chart data)

Regulatory References:
    - IEC 62933 (electrical energy storage systems)
    - UL 9540A (battery safety standard)
    - IEEE 2030.2.1 (battery storage interconnection)

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


class BESSSizingReportTemplate:
    """
    BESS optimization and sizing report template.

    Renders battery energy storage system sizing reports showing
    technology comparisons, dispatch simulation results, degradation
    projections, and financial analysis across markdown, HTML, and
    JSON formats. All outputs include SHA-256 provenance hashing
    for audit trail integrity.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize BESSSizingReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render BESS sizing report as Markdown.

        Args:
            data: BESS sizing engine result data.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_bess_overview(data),
            self._md_technology_comparison(data),
            self._md_dispatch_simulation(data),
            self._md_degradation_projections(data),
            self._md_financial_analysis(data),
            self._md_sensitivity_analysis(data),
            self._md_recommendations(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._generate_provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render BESS sizing report as self-contained HTML.

        Args:
            data: BESS sizing engine result data.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_bess_overview(data),
            self._html_technology_comparison(data),
            self._html_dispatch_simulation(data),
            self._html_degradation_projections(data),
            self._html_financial_analysis(data),
            self._html_sensitivity_analysis(data),
            self._html_recommendations(data),
        ])
        prov = self._generate_provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>BESS Sizing Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render BESS sizing report as structured JSON.

        Args:
            data: BESS sizing engine result data.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "bess_sizing_report",
            "version": "38.0.0",
            "generated_at": self.generated_at.isoformat(),
            "bess_overview": self._json_bess_overview(data),
            "technology_comparison": data.get("technology_comparison", []),
            "dispatch_simulation": data.get("dispatch_simulation", []),
            "degradation_projections": data.get("degradation_projections", []),
            "financial_analysis": self._json_financial_analysis(data),
            "sensitivity_analysis": data.get("sensitivity_analysis", []),
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
            f"# BESS Sizing & Optimization Report\n\n"
            f"**Facility:** {facility}  \n"
            f"**Peak Demand:** {self._format_power(data.get('peak_demand_kw', 0))}  \n"
            f"**Target Reduction:** {self._format_power(data.get('target_reduction_kw', 0))}  \n"
            f"**Project Lifetime:** {data.get('project_lifetime_years', 0)} years  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-038 BESSSizingReportTemplate v38.0.0\n\n---"
        )

    def _md_bess_overview(self, data: Dict[str, Any]) -> str:
        """Render BESS overview summary section."""
        overview = data.get("bess_overview", {})
        return (
            "## 1. BESS Overview\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| Recommended Power Rating | {self._format_power(overview.get('recommended_power_kw', 0))} |\n"
            f"| Recommended Energy Capacity | {self._format_energy(overview.get('recommended_energy_kwh', 0))} |\n"
            f"| Duration | {self._fmt(overview.get('duration_hours', 0), 1)} hours |\n"
            f"| Recommended Chemistry | {overview.get('recommended_chemistry', '-')} |\n"
            f"| Round-Trip Efficiency | {self._fmt(overview.get('round_trip_efficiency', 0))}% |\n"
            f"| Annual Cycles | {self._fmt(overview.get('annual_cycles', 0), 0)} |\n"
            f"| Total CAPEX | {self._format_currency(overview.get('total_capex', 0))} |\n"
            f"| NPV (Project Life) | {self._format_currency(overview.get('npv', 0))} |"
        )

    def _md_technology_comparison(self, data: Dict[str, Any]) -> str:
        """Render technology comparison table."""
        techs = data.get("technology_comparison", [])
        if not techs:
            return "## 2. Technology Comparison\n\n_No technology comparison data available._"
        lines = [
            "## 2. Technology Comparison\n",
            "| Chemistry | Power (kW) | Energy (kWh) | Efficiency | CAPEX | Cycles | Score |",
            "|-----------|--------:|----------:|----------:|------:|------:|------:|",
        ]
        for tech in techs:
            lines.append(
                f"| {tech.get('chemistry', '-')} "
                f"| {self._fmt(tech.get('power_kw', 0), 0)} "
                f"| {self._fmt(tech.get('energy_kwh', 0), 0)} "
                f"| {self._fmt(tech.get('efficiency_pct', 0))}% "
                f"| {self._format_currency(tech.get('capex', 0))} "
                f"| {self._fmt(tech.get('cycle_life', 0), 0)} "
                f"| {self._fmt(tech.get('score', 0), 1)}/100 |"
            )
        return "\n".join(lines)

    def _md_dispatch_simulation(self, data: Dict[str, Any]) -> str:
        """Render dispatch simulation results section."""
        sims = data.get("dispatch_simulation", [])
        if not sims:
            return "## 3. Dispatch Simulation Results\n\n_No simulation data available._"
        lines = [
            "## 3. Dispatch Simulation Results\n",
            "| Month | Peak Before (kW) | Peak After (kW) | Reduction (kW) | Cycles | Revenue |",
            "|-------|----------------:|---------------:|-------------:|------:|--------:|",
        ]
        for sim in sims:
            lines.append(
                f"| {sim.get('month', '-')} "
                f"| {self._fmt(sim.get('peak_before_kw', 0), 1)} "
                f"| {self._fmt(sim.get('peak_after_kw', 0), 1)} "
                f"| {self._fmt(sim.get('reduction_kw', 0), 1)} "
                f"| {self._fmt(sim.get('cycles', 0), 1)} "
                f"| {self._format_currency(sim.get('revenue', 0))} |"
            )
        return "\n".join(lines)

    def _md_degradation_projections(self, data: Dict[str, Any]) -> str:
        """Render degradation projections section."""
        projections = data.get("degradation_projections", [])
        if not projections:
            return "## 4. Degradation Projections\n\n_No degradation data available._"
        lines = [
            "## 4. Degradation Projections\n",
            "| Year | SOH (%) | Usable kWh | Cum. Cycles | Capacity Fade |",
            "|-----:|-------:|----------:|----------:|-------------:|",
        ]
        for proj in projections:
            lines.append(
                f"| {proj.get('year', 0)} "
                f"| {self._fmt(proj.get('soh_pct', 0), 1)} "
                f"| {self._fmt(proj.get('usable_kwh', 0), 0)} "
                f"| {self._fmt(proj.get('cumulative_cycles', 0), 0)} "
                f"| {self._fmt(proj.get('capacity_fade_pct', 0), 2)}% |"
            )
        return "\n".join(lines)

    def _md_financial_analysis(self, data: Dict[str, Any]) -> str:
        """Render financial analysis section."""
        fin = data.get("financial_analysis", {})
        if not fin:
            return "## 5. Financial Analysis\n\n_No financial data available._"
        return (
            "## 5. Financial Analysis\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| Total CAPEX | {self._format_currency(fin.get('total_capex', 0))} |\n"
            f"| Annual OPEX | {self._format_currency(fin.get('annual_opex', 0))} |\n"
            f"| Annual Revenue (Demand Savings) | {self._format_currency(fin.get('annual_demand_savings', 0))} |\n"
            f"| Annual Revenue (Arbitrage) | {self._format_currency(fin.get('annual_arbitrage', 0))} |\n"
            f"| Annual Revenue (Ancillary) | {self._format_currency(fin.get('annual_ancillary', 0))} |\n"
            f"| Net Present Value (NPV) | {self._format_currency(fin.get('npv', 0))} |\n"
            f"| Internal Rate of Return (IRR) | {self._fmt(fin.get('irr_pct', 0))}% |\n"
            f"| Simple Payback | {self._fmt(fin.get('payback_years', 0), 1)} years |"
        )

    def _md_sensitivity_analysis(self, data: Dict[str, Any]) -> str:
        """Render sensitivity analysis section."""
        analysis = data.get("sensitivity_analysis", [])
        if not analysis:
            return "## 6. Sensitivity Analysis\n\n_No sensitivity data available._"
        lines = [
            "## 6. Sensitivity Analysis\n",
            "| Parameter | Low Case | Base Case | High Case | NPV Impact |",
            "|-----------|---------|----------|----------|----------:|",
        ]
        for item in analysis:
            lines.append(
                f"| {item.get('parameter', '-')} "
                f"| {item.get('low_case', '-')} "
                f"| {item.get('base_case', '-')} "
                f"| {item.get('high_case', '-')} "
                f"| {self._format_currency(item.get('npv_impact', 0))} |"
            )
        return "\n".join(lines)

    def _md_recommendations(self, data: Dict[str, Any]) -> str:
        """Render recommendations section."""
        recs = data.get("recommendations", [])
        if not recs:
            recs = [
                "Deploy recommended BESS configuration for optimal peak shaving",
                "Implement revenue stacking with ancillary services",
                "Schedule augmentation at year 7 to maintain capacity targets",
                "Monitor degradation quarterly to optimize dispatch strategy",
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
            f'<h1>BESS Sizing & Optimization Report</h1>\n'
            f'<p class="subtitle">Facility: {facility} | '
            f'Peak Demand: {self._format_power(data.get("peak_demand_kw", 0))} | '
            f'Target: {self._format_power(data.get("target_reduction_kw", 0))}</p>'
        )

    def _html_bess_overview(self, data: Dict[str, Any]) -> str:
        """Render HTML BESS overview cards."""
        o = data.get("bess_overview", {})
        return (
            '<h2>BESS Overview</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Power Rating</span>'
            f'<span class="value">{self._fmt(o.get("recommended_power_kw", 0), 0)} kW</span></div>\n'
            f'  <div class="card"><span class="label">Energy Capacity</span>'
            f'<span class="value">{self._fmt(o.get("recommended_energy_kwh", 0), 0)} kWh</span></div>\n'
            f'  <div class="card"><span class="label">Chemistry</span>'
            f'<span class="value">{o.get("recommended_chemistry", "-")}</span></div>\n'
            f'  <div class="card"><span class="label">Total CAPEX</span>'
            f'<span class="value">{self._format_currency(o.get("total_capex", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">NPV</span>'
            f'<span class="value">{self._format_currency(o.get("npv", 0))}</span></div>\n'
            '</div>'
        )

    def _html_technology_comparison(self, data: Dict[str, Any]) -> str:
        """Render HTML technology comparison table."""
        techs = data.get("technology_comparison", [])
        rows = ""
        for tech in techs:
            rows += (
                f'<tr><td>{tech.get("chemistry", "-")}</td>'
                f'<td>{self._fmt(tech.get("power_kw", 0), 0)}</td>'
                f'<td>{self._fmt(tech.get("energy_kwh", 0), 0)}</td>'
                f'<td>{self._fmt(tech.get("efficiency_pct", 0))}%</td>'
                f'<td>{self._format_currency(tech.get("capex", 0))}</td>'
                f'<td>{self._fmt(tech.get("score", 0), 1)}</td></tr>\n'
            )
        return (
            '<h2>Technology Comparison</h2>\n'
            '<table>\n<tr><th>Chemistry</th><th>Power (kW)</th><th>Energy (kWh)</th>'
            f'<th>Efficiency</th><th>CAPEX</th><th>Score</th></tr>\n{rows}</table>'
        )

    def _html_dispatch_simulation(self, data: Dict[str, Any]) -> str:
        """Render HTML dispatch simulation table."""
        sims = data.get("dispatch_simulation", [])
        rows = ""
        for sim in sims:
            rows += (
                f'<tr><td>{sim.get("month", "-")}</td>'
                f'<td>{self._fmt(sim.get("peak_before_kw", 0), 1)}</td>'
                f'<td>{self._fmt(sim.get("peak_after_kw", 0), 1)}</td>'
                f'<td>{self._fmt(sim.get("reduction_kw", 0), 1)}</td>'
                f'<td>{self._format_currency(sim.get("revenue", 0))}</td></tr>\n'
            )
        return (
            '<h2>Dispatch Simulation</h2>\n'
            '<table>\n<tr><th>Month</th><th>Peak Before</th><th>Peak After</th>'
            f'<th>Reduction</th><th>Revenue</th></tr>\n{rows}</table>'
        )

    def _html_degradation_projections(self, data: Dict[str, Any]) -> str:
        """Render HTML degradation projections table."""
        projs = data.get("degradation_projections", [])
        rows = ""
        for proj in projs:
            rows += (
                f'<tr><td>{proj.get("year", 0)}</td>'
                f'<td>{self._fmt(proj.get("soh_pct", 0), 1)}%</td>'
                f'<td>{self._fmt(proj.get("usable_kwh", 0), 0)}</td>'
                f'<td>{self._fmt(proj.get("cumulative_cycles", 0), 0)}</td>'
                f'<td>{self._fmt(proj.get("capacity_fade_pct", 0), 2)}%</td></tr>\n'
            )
        return (
            '<h2>Degradation Projections</h2>\n'
            '<table>\n<tr><th>Year</th><th>SOH</th><th>Usable kWh</th>'
            f'<th>Cum. Cycles</th><th>Capacity Fade</th></tr>\n{rows}</table>'
        )

    def _html_financial_analysis(self, data: Dict[str, Any]) -> str:
        """Render HTML financial analysis summary."""
        f = data.get("financial_analysis", {})
        return (
            '<h2>Financial Analysis</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Total CAPEX</span>'
            f'<span class="value">{self._format_currency(f.get("total_capex", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">NPV</span>'
            f'<span class="value">{self._format_currency(f.get("npv", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">IRR</span>'
            f'<span class="value">{self._fmt(f.get("irr_pct", 0))}%</span></div>\n'
            f'  <div class="card"><span class="label">Payback</span>'
            f'<span class="value">{self._fmt(f.get("payback_years", 0), 1)} yrs</span></div>\n'
            '</div>'
        )

    def _html_sensitivity_analysis(self, data: Dict[str, Any]) -> str:
        """Render HTML sensitivity analysis table."""
        analysis = data.get("sensitivity_analysis", [])
        rows = ""
        for item in analysis:
            rows += (
                f'<tr><td>{item.get("parameter", "-")}</td>'
                f'<td>{item.get("low_case", "-")}</td>'
                f'<td>{item.get("base_case", "-")}</td>'
                f'<td>{item.get("high_case", "-")}</td>'
                f'<td>{self._format_currency(item.get("npv_impact", 0))}</td></tr>\n'
            )
        return (
            '<h2>Sensitivity Analysis</h2>\n'
            '<table>\n<tr><th>Parameter</th><th>Low</th><th>Base</th>'
            f'<th>High</th><th>NPV Impact</th></tr>\n{rows}</table>'
        )

    def _html_recommendations(self, data: Dict[str, Any]) -> str:
        """Render HTML recommendations."""
        recs = data.get("recommendations", [
            "Deploy recommended BESS configuration",
            "Implement revenue stacking with ancillary services",
        ])
        items = "".join(f'<li>{r}</li>\n' for r in recs)
        return f'<h2>Recommendations</h2>\n<ol>\n{items}</ol>'

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_bess_overview(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON BESS overview."""
        o = data.get("bess_overview", {})
        return {
            "recommended_power_kw": o.get("recommended_power_kw", 0),
            "recommended_energy_kwh": o.get("recommended_energy_kwh", 0),
            "duration_hours": o.get("duration_hours", 0),
            "recommended_chemistry": o.get("recommended_chemistry", ""),
            "round_trip_efficiency": o.get("round_trip_efficiency", 0),
            "annual_cycles": o.get("annual_cycles", 0),
            "total_capex": o.get("total_capex", 0),
            "npv": o.get("npv", 0),
        }

    def _json_financial_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON financial analysis."""
        f = data.get("financial_analysis", {})
        return {
            "total_capex": f.get("total_capex", 0),
            "annual_opex": f.get("annual_opex", 0),
            "annual_demand_savings": f.get("annual_demand_savings", 0),
            "annual_arbitrage": f.get("annual_arbitrage", 0),
            "annual_ancillary": f.get("annual_ancillary", 0),
            "npv": f.get("npv", 0),
            "irr_pct": f.get("irr_pct", 0),
            "payback_years": f.get("payback_years", 0),
        }

    def _json_charts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chart data payloads for visualization."""
        techs = data.get("technology_comparison", [])
        sims = data.get("dispatch_simulation", [])
        degs = data.get("degradation_projections", [])
        sens = data.get("sensitivity_analysis", [])
        return {
            "technology_radar": {
                "type": "radar",
                "labels": [t.get("chemistry", "") for t in techs],
                "values": [t.get("score", 0) for t in techs],
            },
            "dispatch_timeline": {
                "type": "grouped_bar",
                "labels": [s.get("month", "") for s in sims],
                "series": {
                    "before": [s.get("peak_before_kw", 0) for s in sims],
                    "after": [s.get("peak_after_kw", 0) for s in sims],
                },
            },
            "degradation_curve": {
                "type": "line",
                "labels": [str(d.get("year", 0)) for d in degs],
                "values": [d.get("soh_pct", 0) for d in degs],
            },
            "sensitivity_tornado": {
                "type": "tornado",
                "items": [
                    {"label": s.get("parameter", ""), "value": s.get("npv_impact", 0)}
                    for s in sens
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
            val: Energy value in kWh or MWh.

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
