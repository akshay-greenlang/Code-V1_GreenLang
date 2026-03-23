# -*- coding: utf-8 -*-
"""
EnergyConsumptionReportTemplate - Consumption dashboard for PACK-039.

Generates comprehensive energy consumption dashboard reports showing
consumption trends over time, breakdown by system/building/fuel type,
load profile analysis with peak/base/shoulder periods, weather-normalized
consumption overlay, and efficiency ratio tracking.

Sections:
    1. Consumption Overview
    2. Trend Analysis
    3. Breakdown by System
    4. Breakdown by Fuel Type
    5. Load Profile
    6. Weather Overlay
    7. Recommendations

Output Formats:
    - Markdown (with provenance hash comment)
    - HTML (self-contained with inline CSS)
    - JSON (structured with chart data)

Regulatory References:
    - ISO 50001:2018 (Energy management systems)
    - IPMVP Volume I (Measurement and verification)
    - EN 16247 (Energy audits)

Author: GreenLang Team
Version: 39.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODULE_VERSION = "1.0.0"


class EnergyConsumptionReportTemplate:
    """
    Energy consumption dashboard report template.

    Renders consumption dashboard reports showing trend analysis,
    system/building/fuel breakdowns, load profiles with time-of-use
    periods, weather-normalized overlays, and efficiency ratio tracking
    across markdown, HTML, and JSON formats. All outputs include SHA-256
    provenance hashing for audit trail integrity.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize EnergyConsumptionReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render energy consumption report as Markdown.

        Args:
            data: Energy consumption engine result data.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_consumption_overview(data),
            self._md_trend_analysis(data),
            self._md_breakdown_by_system(data),
            self._md_breakdown_by_fuel(data),
            self._md_load_profile(data),
            self._md_weather_overlay(data),
            self._md_recommendations(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._generate_provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render energy consumption report as self-contained HTML.

        Args:
            data: Energy consumption engine result data.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_consumption_overview(data),
            self._html_trend_analysis(data),
            self._html_breakdown_by_system(data),
            self._html_breakdown_by_fuel(data),
            self._html_load_profile(data),
            self._html_weather_overlay(data),
            self._html_recommendations(data),
        ])
        prov = self._generate_provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Energy Consumption Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render energy consumption report as structured JSON.

        Args:
            data: Energy consumption engine result data.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "energy_consumption_report",
            "version": "39.0.0",
            "generated_at": self.generated_at.isoformat(),
            "consumption_overview": self._json_consumption_overview(data),
            "trend_analysis": data.get("trend_analysis", []),
            "breakdown_by_system": data.get("breakdown_by_system", []),
            "breakdown_by_fuel": data.get("breakdown_by_fuel", []),
            "load_profile": data.get("load_profile", []),
            "weather_overlay": data.get("weather_overlay", []),
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
            f"# Energy Consumption Report\n\n"
            f"**Facility:** {facility}  \n"
            f"**Analysis Period:** {data.get('analysis_period', '')}  \n"
            f"**Total Consumption:** {self._format_energy(data.get('total_consumption_mwh', 0))}  \n"
            f"**Peak Demand:** {self._format_power(data.get('peak_demand_kw', 0))}  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-039 EnergyConsumptionReportTemplate v39.0.0\n\n---"
        )

    def _md_consumption_overview(self, data: Dict[str, Any]) -> str:
        """Render consumption overview section."""
        overview = data.get("consumption_overview", {})
        total = data.get("total_consumption_mwh", 0)
        cost = overview.get("total_cost", 0)
        return (
            "## 1. Consumption Overview\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| Total Consumption | {self._format_energy(total)} |\n"
            f"| Total Cost | {self._format_currency(cost)} |\n"
            f"| Average Unit Cost | {self._format_currency(overview.get('avg_unit_cost', 0))}/MWh |\n"
            f"| Peak Demand | {self._format_power(data.get('peak_demand_kw', 0))} |\n"
            f"| Average Demand | {self._format_power(overview.get('avg_demand_kw', 0))} |\n"
            f"| Load Factor | {self._fmt(overview.get('load_factor', 0))}% |\n"
            f"| YoY Change | {self._fmt(overview.get('yoy_change_pct', 0))}% |\n"
            f"| Carbon Intensity | {self._fmt(overview.get('carbon_intensity_kgco2_mwh', 0), 1)} kgCO2/MWh |"
        )

    def _md_trend_analysis(self, data: Dict[str, Any]) -> str:
        """Render trend analysis section."""
        trends = data.get("trend_analysis", [])
        if not trends:
            return "## 2. Trend Analysis\n\n_No trend data available._"
        lines = [
            "## 2. Trend Analysis\n",
            "| Period | Consumption (MWh) | Cost | Peak kW | Load Factor | vs Prev |",
            "|--------|------------------:|-----:|--------:|----------:|--------:|",
        ]
        for t in trends:
            lines.append(
                f"| {t.get('period', '-')} "
                f"| {self._fmt(t.get('consumption_mwh', 0), 1)} "
                f"| {self._format_currency(t.get('cost', 0))} "
                f"| {self._fmt(t.get('peak_kw', 0), 1)} "
                f"| {self._fmt(t.get('load_factor', 0))}% "
                f"| {self._fmt(t.get('vs_prev_pct', 0))}% |"
            )
        return "\n".join(lines)

    def _md_breakdown_by_system(self, data: Dict[str, Any]) -> str:
        """Render breakdown by system section."""
        systems = data.get("breakdown_by_system", [])
        if not systems:
            return "## 3. Breakdown by System\n\n_No system breakdown data available._"
        total = data.get("total_consumption_mwh", 1)
        lines = [
            "## 3. Breakdown by System\n",
            "| System | Consumption (MWh) | % of Total | Cost | EUI (kWh/m2) |",
            "|--------|------------------:|----------:|-----:|------------:|",
        ]
        for s in systems:
            cons = s.get("consumption_mwh", 0)
            lines.append(
                f"| {s.get('system', '-')} "
                f"| {self._fmt(cons, 1)} "
                f"| {self._pct(cons, total)} "
                f"| {self._format_currency(s.get('cost', 0))} "
                f"| {self._fmt(s.get('eui_kwh_m2', 0), 1)} |"
            )
        return "\n".join(lines)

    def _md_breakdown_by_fuel(self, data: Dict[str, Any]) -> str:
        """Render breakdown by fuel type section."""
        fuels = data.get("breakdown_by_fuel", [])
        if not fuels:
            return "## 4. Breakdown by Fuel Type\n\n_No fuel breakdown data available._"
        total = data.get("total_consumption_mwh", 1)
        lines = [
            "## 4. Breakdown by Fuel Type\n",
            "| Fuel Type | Consumption (MWh) | % of Total | Cost | Emission Factor |",
            "|-----------|------------------:|----------:|-----:|---------------:|",
        ]
        for f in fuels:
            cons = f.get("consumption_mwh", 0)
            lines.append(
                f"| {f.get('fuel_type', '-')} "
                f"| {self._fmt(cons, 1)} "
                f"| {self._pct(cons, total)} "
                f"| {self._format_currency(f.get('cost', 0))} "
                f"| {self._fmt(f.get('emission_factor', 0), 3)} tCO2/MWh |"
            )
        return "\n".join(lines)

    def _md_load_profile(self, data: Dict[str, Any]) -> str:
        """Render load profile section."""
        profile = data.get("load_profile", [])
        if not profile:
            return "## 5. Load Profile\n\n_No load profile data available._"
        lines = [
            "## 5. Load Profile\n",
            "| Period | Avg Demand (kW) | Peak Demand (kW) | Consumption (MWh) | Load Factor |",
            "|--------|---------------:|----------------:|------------------:|----------:|",
        ]
        for p in profile:
            lines.append(
                f"| {p.get('period', '-')} "
                f"| {self._fmt(p.get('avg_demand_kw', 0), 1)} "
                f"| {self._fmt(p.get('peak_demand_kw', 0), 1)} "
                f"| {self._fmt(p.get('consumption_mwh', 0), 1)} "
                f"| {self._fmt(p.get('load_factor', 0))}% |"
            )
        return "\n".join(lines)

    def _md_weather_overlay(self, data: Dict[str, Any]) -> str:
        """Render weather overlay section."""
        weather = data.get("weather_overlay", [])
        if not weather:
            return "## 6. Weather Overlay\n\n_No weather overlay data available._"
        lines = [
            "## 6. Weather Overlay\n",
            "| Period | HDD | CDD | Actual (MWh) | Normalized (MWh) | Weather Impact (%) |",
            "|--------|----:|----:|-----------:|-----------------:|------------------:|",
        ]
        for w in weather:
            lines.append(
                f"| {w.get('period', '-')} "
                f"| {self._fmt(w.get('hdd', 0), 0)} "
                f"| {self._fmt(w.get('cdd', 0), 0)} "
                f"| {self._fmt(w.get('actual_mwh', 0), 1)} "
                f"| {self._fmt(w.get('normalized_mwh', 0), 1)} "
                f"| {self._fmt(w.get('weather_impact_pct', 0))}% |"
            )
        return "\n".join(lines)

    def _md_recommendations(self, data: Dict[str, Any]) -> str:
        """Render recommendations section."""
        recs = data.get("recommendations", [])
        if not recs:
            recs = [
                "Investigate high-consumption systems for efficiency improvements",
                "Shift flexible loads to off-peak periods to reduce demand charges",
                "Implement weather-normalized tracking for accurate performance baselining",
                "Review fuel mix for carbon reduction opportunities",
            ]
        lines = ["## 7. Recommendations\n"]
        for i, rec in enumerate(recs, 1):
            lines.append(f"{i}. {rec}")
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render report footer."""
        return "---\n\n*Generated by GreenLang PACK-039 Energy Monitoring Pack*"

    # ------------------------------------------------------------------
    # HTML section renderers
    # ------------------------------------------------------------------

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        facility = data.get("facility_name", "Facility")
        return (
            f'<h1>Energy Consumption Report</h1>\n'
            f'<p class="subtitle">Facility: {facility} | '
            f'Total: {self._format_energy(data.get("total_consumption_mwh", 0))} | '
            f'Period: {data.get("analysis_period", "-")}</p>'
        )

    def _html_consumption_overview(self, data: Dict[str, Any]) -> str:
        """Render HTML consumption overview cards."""
        o = data.get("consumption_overview", {})
        total = data.get("total_consumption_mwh", 0)
        return (
            '<h2>Consumption Overview</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Total Consumption</span>'
            f'<span class="value">{self._fmt(total, 1)} MWh</span></div>\n'
            f'  <div class="card"><span class="label">Total Cost</span>'
            f'<span class="value">{self._format_currency(o.get("total_cost", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Peak Demand</span>'
            f'<span class="value">{self._fmt(data.get("peak_demand_kw", 0), 0)} kW</span></div>\n'
            f'  <div class="card"><span class="label">Load Factor</span>'
            f'<span class="value">{self._fmt(o.get("load_factor", 0))}%</span></div>\n'
            f'  <div class="card"><span class="label">YoY Change</span>'
            f'<span class="value">{self._fmt(o.get("yoy_change_pct", 0))}%</span></div>\n'
            '</div>'
        )

    def _html_trend_analysis(self, data: Dict[str, Any]) -> str:
        """Render HTML trend analysis table."""
        trends = data.get("trend_analysis", [])
        rows = ""
        for t in trends:
            rows += (
                f'<tr><td>{t.get("period", "-")}</td>'
                f'<td>{self._fmt(t.get("consumption_mwh", 0), 1)}</td>'
                f'<td>{self._format_currency(t.get("cost", 0))}</td>'
                f'<td>{self._fmt(t.get("peak_kw", 0), 1)}</td>'
                f'<td>{self._fmt(t.get("load_factor", 0))}%</td>'
                f'<td>{self._fmt(t.get("vs_prev_pct", 0))}%</td></tr>\n'
            )
        return (
            '<h2>Trend Analysis</h2>\n'
            '<table>\n<tr><th>Period</th><th>Consumption (MWh)</th><th>Cost</th>'
            f'<th>Peak kW</th><th>Load Factor</th><th>vs Prev</th></tr>\n{rows}</table>'
        )

    def _html_breakdown_by_system(self, data: Dict[str, Any]) -> str:
        """Render HTML system breakdown table."""
        systems = data.get("breakdown_by_system", [])
        total = data.get("total_consumption_mwh", 1)
        rows = ""
        for s in systems:
            cons = s.get("consumption_mwh", 0)
            rows += (
                f'<tr><td>{s.get("system", "-")}</td>'
                f'<td>{self._fmt(cons, 1)}</td>'
                f'<td>{self._pct(cons, total)}</td>'
                f'<td>{self._format_currency(s.get("cost", 0))}</td>'
                f'<td>{self._fmt(s.get("eui_kwh_m2", 0), 1)}</td></tr>\n'
            )
        return (
            '<h2>Breakdown by System</h2>\n'
            '<table>\n<tr><th>System</th><th>Consumption (MWh)</th><th>% of Total</th>'
            f'<th>Cost</th><th>EUI (kWh/m2)</th></tr>\n{rows}</table>'
        )

    def _html_breakdown_by_fuel(self, data: Dict[str, Any]) -> str:
        """Render HTML fuel breakdown table."""
        fuels = data.get("breakdown_by_fuel", [])
        total = data.get("total_consumption_mwh", 1)
        rows = ""
        for f in fuels:
            cons = f.get("consumption_mwh", 0)
            rows += (
                f'<tr><td>{f.get("fuel_type", "-")}</td>'
                f'<td>{self._fmt(cons, 1)}</td>'
                f'<td>{self._pct(cons, total)}</td>'
                f'<td>{self._format_currency(f.get("cost", 0))}</td>'
                f'<td>{self._fmt(f.get("emission_factor", 0), 3)}</td></tr>\n'
            )
        return (
            '<h2>Breakdown by Fuel Type</h2>\n'
            '<table>\n<tr><th>Fuel Type</th><th>Consumption (MWh)</th><th>% of Total</th>'
            f'<th>Cost</th><th>Emission Factor (tCO2/MWh)</th></tr>\n{rows}</table>'
        )

    def _html_load_profile(self, data: Dict[str, Any]) -> str:
        """Render HTML load profile table."""
        profile = data.get("load_profile", [])
        rows = ""
        for p in profile:
            rows += (
                f'<tr><td>{p.get("period", "-")}</td>'
                f'<td>{self._fmt(p.get("avg_demand_kw", 0), 1)}</td>'
                f'<td>{self._fmt(p.get("peak_demand_kw", 0), 1)}</td>'
                f'<td>{self._fmt(p.get("consumption_mwh", 0), 1)}</td>'
                f'<td>{self._fmt(p.get("load_factor", 0))}%</td></tr>\n'
            )
        return (
            '<h2>Load Profile</h2>\n'
            '<table>\n<tr><th>Period</th><th>Avg kW</th><th>Peak kW</th>'
            f'<th>Consumption (MWh)</th><th>Load Factor</th></tr>\n{rows}</table>'
        )

    def _html_weather_overlay(self, data: Dict[str, Any]) -> str:
        """Render HTML weather overlay table."""
        weather = data.get("weather_overlay", [])
        rows = ""
        for w in weather:
            rows += (
                f'<tr><td>{w.get("period", "-")}</td>'
                f'<td>{self._fmt(w.get("hdd", 0), 0)}</td>'
                f'<td>{self._fmt(w.get("cdd", 0), 0)}</td>'
                f'<td>{self._fmt(w.get("actual_mwh", 0), 1)}</td>'
                f'<td>{self._fmt(w.get("normalized_mwh", 0), 1)}</td>'
                f'<td>{self._fmt(w.get("weather_impact_pct", 0))}%</td></tr>\n'
            )
        return (
            '<h2>Weather Overlay</h2>\n'
            '<table>\n<tr><th>Period</th><th>HDD</th><th>CDD</th>'
            '<th>Actual (MWh)</th><th>Normalized (MWh)</th>'
            f'<th>Weather Impact (%)</th></tr>\n{rows}</table>'
        )

    def _html_recommendations(self, data: Dict[str, Any]) -> str:
        """Render HTML recommendations."""
        recs = data.get("recommendations", [
            "Investigate high-consumption systems for efficiency improvements",
            "Shift flexible loads to off-peak periods to reduce demand charges",
        ])
        items = "".join(f'<li>{r}</li>\n' for r in recs)
        return f'<h2>Recommendations</h2>\n<ol>\n{items}</ol>'

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_consumption_overview(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON consumption overview."""
        o = data.get("consumption_overview", {})
        return {
            "total_consumption_mwh": data.get("total_consumption_mwh", 0),
            "total_cost": o.get("total_cost", 0),
            "avg_unit_cost": o.get("avg_unit_cost", 0),
            "peak_demand_kw": data.get("peak_demand_kw", 0),
            "avg_demand_kw": o.get("avg_demand_kw", 0),
            "load_factor": o.get("load_factor", 0),
            "yoy_change_pct": o.get("yoy_change_pct", 0),
            "carbon_intensity_kgco2_mwh": o.get("carbon_intensity_kgco2_mwh", 0),
        }

    def _json_charts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chart data payloads for visualization."""
        trends = data.get("trend_analysis", [])
        systems = data.get("breakdown_by_system", [])
        fuels = data.get("breakdown_by_fuel", [])
        weather = data.get("weather_overlay", [])
        return {
            "consumption_trend": {
                "type": "line",
                "labels": [t.get("period", "") for t in trends],
                "values": [t.get("consumption_mwh", 0) for t in trends],
            },
            "system_breakdown": {
                "type": "pie",
                "labels": [s.get("system", "") for s in systems],
                "values": [s.get("consumption_mwh", 0) for s in systems],
            },
            "fuel_breakdown": {
                "type": "pie",
                "labels": [f.get("fuel_type", "") for f in fuels],
                "values": [f.get("consumption_mwh", 0) for f in fuels],
            },
            "weather_comparison": {
                "type": "dual_axis",
                "labels": [w.get("period", "") for w in weather],
                "series": {
                    "actual": [w.get("actual_mwh", 0) for w in weather],
                    "normalized": [w.get("normalized_mwh", 0) for w in weather],
                    "hdd": [w.get("hdd", 0) for w in weather],
                    "cdd": [w.get("cdd", 0) for w in weather],
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
