# -*- coding: utf-8 -*-
"""
LoadShiftingReportTemplate - Load shifting plan for PACK-038.

Generates comprehensive load shifting plan reports showing shiftable
load inventory with flexibility windows, constraint summaries for each
load including process dependencies and safety limits, optimized
dispatch schedules, and rebound effect analysis with mitigation
strategies.

Sections:
    1. Shifting Summary
    2. Shiftable Load Inventory
    3. Constraint Summary
    4. Optimized Schedule
    5. Rebound Analysis
    6. Cost Impact
    7. Recommendations

Output Formats:
    - Markdown (with provenance hash comment)
    - HTML (self-contained with inline CSS)
    - JSON (structured with chart data)

Regulatory References:
    - EN 15232 (building automation impact on energy)
    - IEC 61850 (substation automation for load control)
    - ISO 50001 (energy management systems)

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


class LoadShiftingReportTemplate:
    """
    Load shifting plan report template.

    Renders load shifting plan reports showing shiftable load inventory,
    constraint summaries, optimized dispatch schedules, and rebound
    analysis across markdown, HTML, and JSON formats. All outputs
    include SHA-256 provenance hashing for audit trail integrity.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize LoadShiftingReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render load shifting report as Markdown.

        Args:
            data: Load shifting engine result data.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_shifting_summary(data),
            self._md_shiftable_inventory(data),
            self._md_constraint_summary(data),
            self._md_optimized_schedule(data),
            self._md_rebound_analysis(data),
            self._md_cost_impact(data),
            self._md_recommendations(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._generate_provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render load shifting report as self-contained HTML.

        Args:
            data: Load shifting engine result data.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_shifting_summary(data),
            self._html_shiftable_inventory(data),
            self._html_constraint_summary(data),
            self._html_optimized_schedule(data),
            self._html_rebound_analysis(data),
            self._html_cost_impact(data),
            self._html_recommendations(data),
        ])
        prov = self._generate_provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Load Shifting Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render load shifting report as structured JSON.

        Args:
            data: Load shifting engine result data.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "load_shifting_report",
            "version": "38.0.0",
            "generated_at": self.generated_at.isoformat(),
            "shifting_summary": self._json_shifting_summary(data),
            "shiftable_inventory": data.get("shiftable_inventory", []),
            "constraint_summary": data.get("constraint_summary", []),
            "optimized_schedule": data.get("optimized_schedule", []),
            "rebound_analysis": data.get("rebound_analysis", []),
            "cost_impact": self._json_cost_impact(data),
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
            f"# Load Shifting Plan Report\n\n"
            f"**Facility:** {facility}  \n"
            f"**Peak Demand:** {self._format_power(data.get('peak_demand_kw', 0))}  \n"
            f"**Shiftable Capacity:** {self._format_power(data.get('shiftable_capacity_kw', 0))}  \n"
            f"**Optimization Window:** {data.get('optimization_window', '')}  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-038 LoadShiftingReportTemplate v38.0.0\n\n---"
        )

    def _md_shifting_summary(self, data: Dict[str, Any]) -> str:
        """Render shifting summary section."""
        summary = data.get("shifting_summary", {})
        return (
            "## 1. Shifting Summary\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| Total Shiftable Loads | {summary.get('total_shiftable_loads', 0)} |\n"
            f"| Shiftable Capacity | {self._format_power(summary.get('shiftable_capacity_kw', 0))} |\n"
            f"| Peak Reduction Achieved | {self._format_power(summary.get('peak_reduction_kw', 0))} |\n"
            f"| Energy Shifted Daily | {self._format_energy(summary.get('energy_shifted_kwh', 0))} |\n"
            f"| Avg Shift Duration | {self._fmt(summary.get('avg_shift_hours', 0), 1)} hours |\n"
            f"| Rebound Factor | {self._fmt(summary.get('rebound_factor', 0), 2)} |\n"
            f"| Monthly Cost Savings | {self._format_currency(summary.get('monthly_savings', 0))} |\n"
            f"| Comfort Impact Score | {self._fmt(summary.get('comfort_impact_score', 0), 1)}/10 |"
        )

    def _md_shiftable_inventory(self, data: Dict[str, Any]) -> str:
        """Render shiftable load inventory table."""
        loads = data.get("shiftable_inventory", [])
        if not loads:
            return "## 2. Shiftable Load Inventory\n\n_No shiftable load data available._"
        lines = [
            "## 2. Shiftable Load Inventory\n",
            "| # | Load Name | Rated kW | Shift Window | Max Shift (hrs) | Priority |",
            "|---|-----------|-------:|-------------|-------------:|----------|",
        ]
        for i, load in enumerate(loads, 1):
            lines.append(
                f"| {i} | {load.get('name', '-')} "
                f"| {self._fmt(load.get('rated_kw', 0), 1)} "
                f"| {load.get('shift_window', '-')} "
                f"| {self._fmt(load.get('max_shift_hours', 0), 1)} "
                f"| {load.get('priority', '-')} |"
            )
        return "\n".join(lines)

    def _md_constraint_summary(self, data: Dict[str, Any]) -> str:
        """Render constraint summary section."""
        constraints = data.get("constraint_summary", [])
        if not constraints:
            return "## 3. Constraint Summary\n\n_No constraint data available._"
        lines = [
            "## 3. Constraint Summary\n",
            "| Load | Constraint Type | Description | Severity | Mitigation |",
            "|------|----------------|-------------|----------|------------|",
        ]
        for c in constraints:
            lines.append(
                f"| {c.get('load_name', '-')} "
                f"| {c.get('constraint_type', '-')} "
                f"| {c.get('description', '-')} "
                f"| {c.get('severity', '-')} "
                f"| {c.get('mitigation', '-')} |"
            )
        return "\n".join(lines)

    def _md_optimized_schedule(self, data: Dict[str, Any]) -> str:
        """Render optimized schedule section."""
        schedule = data.get("optimized_schedule", [])
        if not schedule:
            return "## 4. Optimized Schedule\n\n_No schedule data available._"
        lines = [
            "## 4. Optimized Schedule\n",
            "| Time Block | Action | Load(s) | kW Change | New Demand (kW) |",
            "|-----------|--------|---------|--------:|---------------:|",
        ]
        for entry in schedule:
            loads_str = ", ".join(entry.get("loads", ["-"])[:3])
            lines.append(
                f"| {entry.get('time_block', '-')} "
                f"| {entry.get('action', '-')} "
                f"| {loads_str} "
                f"| {self._fmt(entry.get('kw_change', 0), 1)} "
                f"| {self._fmt(entry.get('new_demand_kw', 0), 1)} |"
            )
        return "\n".join(lines)

    def _md_rebound_analysis(self, data: Dict[str, Any]) -> str:
        """Render rebound analysis section."""
        rebounds = data.get("rebound_analysis", [])
        if not rebounds:
            return "## 5. Rebound Analysis\n\n_No rebound data available._"
        lines = [
            "## 5. Rebound Analysis\n",
            "| Load | Shift kW | Rebound kW | Rebound Time | Factor | Strategy |",
            "|------|-------:|----------:|-------------|------:|----------|",
        ]
        for r in rebounds:
            lines.append(
                f"| {r.get('load_name', '-')} "
                f"| {self._fmt(r.get('shift_kw', 0), 1)} "
                f"| {self._fmt(r.get('rebound_kw', 0), 1)} "
                f"| {r.get('rebound_time', '-')} "
                f"| {self._fmt(r.get('rebound_factor', 0), 2)} "
                f"| {r.get('mitigation_strategy', '-')} |"
            )
        return "\n".join(lines)

    def _md_cost_impact(self, data: Dict[str, Any]) -> str:
        """Render cost impact section."""
        cost = data.get("cost_impact", {})
        if not cost:
            return "## 6. Cost Impact\n\n_No cost impact data available._"
        return (
            "## 6. Cost Impact\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| Demand Charge Savings | {self._format_currency(cost.get('demand_charge_savings', 0))} |\n"
            f"| Energy Arbitrage Savings | {self._format_currency(cost.get('energy_arbitrage_savings', 0))} |\n"
            f"| Rebound Energy Cost | {self._format_currency(cost.get('rebound_energy_cost', 0))} |\n"
            f"| Net Monthly Savings | {self._format_currency(cost.get('net_monthly_savings', 0))} |\n"
            f"| Net Annual Savings | {self._format_currency(cost.get('net_annual_savings', 0))} |\n"
            f"| Implementation Cost | {self._format_currency(cost.get('implementation_cost', 0))} |\n"
            f"| Simple Payback | {self._fmt(cost.get('payback_months', 0), 0)} months |"
        )

    def _md_recommendations(self, data: Dict[str, Any]) -> str:
        """Render recommendations section."""
        recs = data.get("recommendations", [])
        if not recs:
            recs = [
                "Implement automated load shifting for HVAC pre-cooling",
                "Deploy smart scheduling for process loads during off-peak",
                "Install thermal storage to extend shifting capability",
                "Monitor rebound effects and adjust mitigation strategies",
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
            f'<h1>Load Shifting Plan Report</h1>\n'
            f'<p class="subtitle">Facility: {facility} | '
            f'Peak Demand: {self._format_power(data.get("peak_demand_kw", 0))} | '
            f'Shiftable: {self._format_power(data.get("shiftable_capacity_kw", 0))}</p>'
        )

    def _html_shifting_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML shifting summary cards."""
        s = data.get("shifting_summary", {})
        return (
            '<h2>Shifting Summary</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Shiftable Loads</span>'
            f'<span class="value">{s.get("total_shiftable_loads", 0)}</span></div>\n'
            f'  <div class="card"><span class="label">Shiftable kW</span>'
            f'<span class="value">{self._fmt(s.get("shiftable_capacity_kw", 0), 0)}</span></div>\n'
            f'  <div class="card"><span class="label">Peak Reduction</span>'
            f'<span class="value">{self._fmt(s.get("peak_reduction_kw", 0), 0)} kW</span></div>\n'
            f'  <div class="card"><span class="label">Monthly Savings</span>'
            f'<span class="value">{self._format_currency(s.get("monthly_savings", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Rebound Factor</span>'
            f'<span class="value">{self._fmt(s.get("rebound_factor", 0), 2)}</span></div>\n'
            '</div>'
        )

    def _html_shiftable_inventory(self, data: Dict[str, Any]) -> str:
        """Render HTML shiftable inventory table."""
        loads = data.get("shiftable_inventory", [])
        rows = ""
        for load in loads:
            rows += (
                f'<tr><td>{load.get("name", "-")}</td>'
                f'<td>{self._fmt(load.get("rated_kw", 0), 1)}</td>'
                f'<td>{load.get("shift_window", "-")}</td>'
                f'<td>{self._fmt(load.get("max_shift_hours", 0), 1)}</td>'
                f'<td>{load.get("priority", "-")}</td></tr>\n'
            )
        return (
            '<h2>Shiftable Load Inventory</h2>\n'
            '<table>\n<tr><th>Load</th><th>Rated kW</th><th>Shift Window</th>'
            f'<th>Max Shift (hrs)</th><th>Priority</th></tr>\n{rows}</table>'
        )

    def _html_constraint_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML constraint summary table."""
        constraints = data.get("constraint_summary", [])
        rows = ""
        for c in constraints:
            sev = c.get("severity", "low").lower()
            rows += (
                f'<tr><td>{c.get("load_name", "-")}</td>'
                f'<td>{c.get("constraint_type", "-")}</td>'
                f'<td>{c.get("description", "-")}</td>'
                f'<td class="severity-{sev}">{c.get("severity", "-")}</td>'
                f'<td>{c.get("mitigation", "-")}</td></tr>\n'
            )
        return (
            '<h2>Constraint Summary</h2>\n'
            '<table>\n<tr><th>Load</th><th>Type</th><th>Description</th>'
            f'<th>Severity</th><th>Mitigation</th></tr>\n{rows}</table>'
        )

    def _html_optimized_schedule(self, data: Dict[str, Any]) -> str:
        """Render HTML optimized schedule table."""
        schedule = data.get("optimized_schedule", [])
        rows = ""
        for entry in schedule:
            loads_str = ", ".join(entry.get("loads", ["-"])[:3])
            rows += (
                f'<tr><td>{entry.get("time_block", "-")}</td>'
                f'<td>{entry.get("action", "-")}</td>'
                f'<td>{loads_str}</td>'
                f'<td>{self._fmt(entry.get("kw_change", 0), 1)}</td>'
                f'<td>{self._fmt(entry.get("new_demand_kw", 0), 1)}</td></tr>\n'
            )
        return (
            '<h2>Optimized Schedule</h2>\n'
            '<table>\n<tr><th>Time Block</th><th>Action</th><th>Loads</th>'
            f'<th>kW Change</th><th>New Demand</th></tr>\n{rows}</table>'
        )

    def _html_rebound_analysis(self, data: Dict[str, Any]) -> str:
        """Render HTML rebound analysis table."""
        rebounds = data.get("rebound_analysis", [])
        rows = ""
        for r in rebounds:
            rows += (
                f'<tr><td>{r.get("load_name", "-")}</td>'
                f'<td>{self._fmt(r.get("shift_kw", 0), 1)}</td>'
                f'<td>{self._fmt(r.get("rebound_kw", 0), 1)}</td>'
                f'<td>{r.get("rebound_time", "-")}</td>'
                f'<td>{self._fmt(r.get("rebound_factor", 0), 2)}</td></tr>\n'
            )
        return (
            '<h2>Rebound Analysis</h2>\n'
            '<table>\n<tr><th>Load</th><th>Shift kW</th><th>Rebound kW</th>'
            f'<th>Rebound Time</th><th>Factor</th></tr>\n{rows}</table>'
        )

    def _html_cost_impact(self, data: Dict[str, Any]) -> str:
        """Render HTML cost impact summary."""
        c = data.get("cost_impact", {})
        return (
            '<h2>Cost Impact</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Demand Savings</span>'
            f'<span class="value">{self._format_currency(c.get("demand_charge_savings", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Arbitrage Savings</span>'
            f'<span class="value">{self._format_currency(c.get("energy_arbitrage_savings", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Net Annual</span>'
            f'<span class="value">{self._format_currency(c.get("net_annual_savings", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Payback</span>'
            f'<span class="value">{self._fmt(c.get("payback_months", 0), 0)} months</span></div>\n'
            '</div>'
        )

    def _html_recommendations(self, data: Dict[str, Any]) -> str:
        """Render HTML recommendations."""
        recs = data.get("recommendations", [
            "Implement automated load shifting for HVAC pre-cooling",
            "Deploy smart scheduling for process loads during off-peak",
        ])
        items = "".join(f'<li>{r}</li>\n' for r in recs)
        return f'<h2>Recommendations</h2>\n<ol>\n{items}</ol>'

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_shifting_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON shifting summary."""
        s = data.get("shifting_summary", {})
        return {
            "total_shiftable_loads": s.get("total_shiftable_loads", 0),
            "shiftable_capacity_kw": s.get("shiftable_capacity_kw", 0),
            "peak_reduction_kw": s.get("peak_reduction_kw", 0),
            "energy_shifted_kwh": s.get("energy_shifted_kwh", 0),
            "avg_shift_hours": s.get("avg_shift_hours", 0),
            "rebound_factor": s.get("rebound_factor", 0),
            "monthly_savings": s.get("monthly_savings", 0),
            "comfort_impact_score": s.get("comfort_impact_score", 0),
        }

    def _json_cost_impact(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON cost impact."""
        c = data.get("cost_impact", {})
        return {
            "demand_charge_savings": c.get("demand_charge_savings", 0),
            "energy_arbitrage_savings": c.get("energy_arbitrage_savings", 0),
            "rebound_energy_cost": c.get("rebound_energy_cost", 0),
            "net_monthly_savings": c.get("net_monthly_savings", 0),
            "net_annual_savings": c.get("net_annual_savings", 0),
            "implementation_cost": c.get("implementation_cost", 0),
            "payback_months": c.get("payback_months", 0),
        }

    def _json_charts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chart data payloads for visualization."""
        loads = data.get("shiftable_inventory", [])
        schedule = data.get("optimized_schedule", [])
        rebounds = data.get("rebound_analysis", [])
        return {
            "shiftable_capacity_bar": {
                "type": "bar",
                "labels": [l.get("name", "") for l in loads],
                "values": [l.get("rated_kw", 0) for l in loads],
            },
            "schedule_timeline": {
                "type": "timeline",
                "items": [
                    {
                        "time_block": e.get("time_block", ""),
                        "action": e.get("action", ""),
                        "kw_change": e.get("kw_change", 0),
                    }
                    for e in schedule
                ],
            },
            "demand_profile": {
                "type": "area",
                "labels": [e.get("time_block", "") for e in schedule],
                "values": [e.get("new_demand_kw", 0) for e in schedule],
            },
            "rebound_comparison": {
                "type": "grouped_bar",
                "labels": [r.get("load_name", "") for r in rebounds],
                "series": {
                    "shifted": [r.get("shift_kw", 0) for r in rebounds],
                    "rebound": [r.get("rebound_kw", 0) for r in rebounds],
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
