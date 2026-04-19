# -*- coding: utf-8 -*-
"""
DERPerformanceReportTemplate - DER asset performance during DR for PACK-037.

Generates distributed energy resource performance reports during demand
response events covering battery state-of-charge profiles, solar PV
contribution, EV fleet flexibility, thermal storage utilization, and
cross-asset coordination effectiveness.

Sections:
    1. DER Fleet Summary
    2. Battery Storage Performance
    3. Solar PV Contribution
    4. EV Fleet Flexibility
    5. Thermal Storage Utilization
    6. Cross-Asset Coordination
    7. DER Optimization Recommendations

Output Formats:
    - Markdown (with provenance hash comment)
    - HTML (self-contained with inline CSS)
    - JSON (structured with chart data)

Regulatory References:
    - FERC Order 2222 (DER aggregation)
    - IEEE 2030.5 (DER communication)
    - IEC 61850-7-420 (DER modeling)
    - SAE J3072 (V2G standards)

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


class DERPerformanceReportTemplate:
    """
    DER asset performance report template.

    Renders DER performance analysis during DR events with battery SOC
    profiles, solar contribution, EV flexibility, thermal storage, and
    coordination metrics across markdown, HTML, and JSON formats.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize DERPerformanceReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render DER performance report as Markdown.

        Args:
            data: DER performance engine result data.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_fleet_summary(data),
            self._md_battery_performance(data),
            self._md_solar_contribution(data),
            self._md_ev_flexibility(data),
            self._md_thermal_storage(data),
            self._md_coordination(data),
            self._md_recommendations(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._generate_provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render DER performance report as self-contained HTML.

        Args:
            data: DER performance engine result data.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_fleet_summary(data),
            self._html_battery_performance(data),
            self._html_solar_contribution(data),
            self._html_ev_flexibility(data),
            self._html_thermal_storage(data),
            self._html_coordination(data),
            self._html_recommendations(data),
        ])
        prov = self._generate_provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>DER Performance Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render DER performance report as structured JSON.

        Args:
            data: DER performance engine result data.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "der_performance_report",
            "version": "37.0.0",
            "generated_at": self.generated_at.isoformat(),
            "fleet_summary": self._json_fleet_summary(data),
            "battery_performance": data.get("battery_performance", []),
            "solar_contribution": data.get("solar_contribution", {}),
            "ev_flexibility": data.get("ev_flexibility", {}),
            "thermal_storage": data.get("thermal_storage", {}),
            "coordination": data.get("coordination", {}),
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
        """Render markdown header."""
        facility = data.get("facility_name", "Facility")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# DER Performance Report\n\n"
            f"**Facility:** {facility}  \n"
            f"**Event ID:** {data.get('event_id', '')}  \n"
            f"**Event Date:** {data.get('event_date', '')}  \n"
            f"**DER Assets:** {data.get('total_der_assets', 0)}  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-037 DERPerformanceReportTemplate v37.0.0\n\n---"
        )

    def _md_fleet_summary(self, data: Dict[str, Any]) -> str:
        """Render DER fleet summary section."""
        summary = data.get("fleet_summary", {})
        return (
            "## 1. DER Fleet Summary\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| Total DER Assets | {summary.get('total_assets', 0)} |\n"
            f"| Total Installed Capacity | {self._format_power(summary.get('total_capacity_kw', 0))} |\n"
            f"| Dispatched Capacity | {self._format_power(summary.get('dispatched_capacity_kw', 0))} |\n"
            f"| Delivered Capacity | {self._format_power(summary.get('delivered_capacity_kw', 0))} |\n"
            f"| Fleet Utilization | {self._fmt(summary.get('fleet_utilization_pct', 0))}% |\n"
            f"| Fleet Performance Score | {self._fmt(summary.get('performance_score', 0), 1)}/100 |\n"
            f"| Battery Assets | {summary.get('battery_count', 0)} |\n"
            f"| Solar Assets | {summary.get('solar_count', 0)} |\n"
            f"| EV Chargers | {summary.get('ev_count', 0)} |\n"
            f"| Thermal Storage | {summary.get('thermal_count', 0)} |"
        )

    def _md_battery_performance(self, data: Dict[str, Any]) -> str:
        """Render battery storage performance section."""
        batteries = data.get("battery_performance", [])
        if not batteries:
            return "## 2. Battery Storage Performance\n\n_No battery data available._"
        lines = [
            "## 2. Battery Storage Performance\n",
            "| Battery | Capacity (kWh) | Start SOC (%) | End SOC (%) | Delivered (kW) | Duration | Efficiency (%) |",
            "|---------|---------------:|-------------:|------------:|---------------:|---------:|---------------:|",
        ]
        for b in batteries:
            lines.append(
                f"| {b.get('name', '-')} "
                f"| {self._fmt(b.get('capacity_kwh', 0), 1)} "
                f"| {self._fmt(b.get('start_soc_pct', 0), 1)} "
                f"| {self._fmt(b.get('end_soc_pct', 0), 1)} "
                f"| {self._fmt(b.get('delivered_kw', 0), 1)} "
                f"| {b.get('duration_hours', 0)} hrs "
                f"| {self._fmt(b.get('round_trip_efficiency_pct', 0))} |"
            )
        return "\n".join(lines)

    def _md_solar_contribution(self, data: Dict[str, Any]) -> str:
        """Render solar PV contribution section."""
        solar = data.get("solar_contribution", {})
        return (
            "## 3. Solar PV Contribution\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| Installed Capacity | {self._format_power(solar.get('installed_capacity_kw', 0))} |\n"
            f"| Generation During Event | {self._format_energy(solar.get('generation_mwh', 0))} |\n"
            f"| Avg Output (% of rated) | {self._fmt(solar.get('avg_output_pct', 0))}% |\n"
            f"| Peak Output | {self._format_power(solar.get('peak_output_kw', 0))} |\n"
            f"| Cloud Impact | {solar.get('cloud_impact', '-')} |\n"
            f"| Forecast Accuracy | {self._fmt(solar.get('forecast_accuracy_pct', 0))}% |\n"
            f"| DR Contribution | {self._format_power(solar.get('dr_contribution_kw', 0))} |"
        )

    def _md_ev_flexibility(self, data: Dict[str, Any]) -> str:
        """Render EV fleet flexibility section."""
        ev = data.get("ev_flexibility", {})
        vehicles = ev.get("vehicles", [])
        lines = [
            "## 4. EV Fleet Flexibility\n",
            f"**Total EV Chargers:** {ev.get('total_chargers', 0)}  ",
            f"**Curtailed Charging Load:** {self._format_power(ev.get('curtailed_load_kw', 0))}  ",
            f"**V2G Discharge:** {self._format_power(ev.get('v2g_discharge_kw', 0))}  ",
            f"**EVs Participating:** {ev.get('evs_participating', 0)}\n",
        ]
        if vehicles:
            lines.extend([
                "| Vehicle/Charger | Action | kW Impact | SOC Start | SOC End | Owner Notified |",
                "|-----------------|--------|----------:|----------:|--------:|---------------|",
            ])
            for v in vehicles:
                lines.append(
                    f"| {v.get('name', '-')} "
                    f"| {v.get('action', '-')} "
                    f"| {self._fmt(v.get('kw_impact', 0), 1)} "
                    f"| {self._fmt(v.get('soc_start_pct', 0), 0)}% "
                    f"| {self._fmt(v.get('soc_end_pct', 0), 0)}% "
                    f"| {'Yes' if v.get('owner_notified', False) else 'No'} |"
                )
        return "\n".join(lines)

    def _md_thermal_storage(self, data: Dict[str, Any]) -> str:
        """Render thermal storage utilization section."""
        thermal = data.get("thermal_storage", {})
        assets = thermal.get("assets", [])
        lines = [
            "## 5. Thermal Storage Utilization\n",
            f"**Total Thermal Capacity:** {self._fmt(thermal.get('total_capacity_kwh_th', 0), 0)} kWh-th  ",
            f"**Electrical Load Deferred:** {self._format_power(thermal.get('electrical_load_deferred_kw', 0))}  ",
            f"**Pre-Cooling/Heating Duration:** {thermal.get('pre_conditioning_hours', 0)} hrs\n",
        ]
        if assets:
            lines.extend([
                "| Asset | Type | Capacity (kWh-th) | Load Deferred (kW) | Duration | Comfort Impact |",
                "|-------|------|------------------:|-------------------:|---------:|---------------|",
            ])
            for a in assets:
                lines.append(
                    f"| {a.get('name', '-')} "
                    f"| {a.get('type', '-')} "
                    f"| {self._fmt(a.get('capacity_kwh_th', 0), 0)} "
                    f"| {self._fmt(a.get('load_deferred_kw', 0), 1)} "
                    f"| {a.get('duration_hours', 0)} hrs "
                    f"| {a.get('comfort_impact', '-')} |"
                )
        return "\n".join(lines)

    def _md_coordination(self, data: Dict[str, Any]) -> str:
        """Render cross-asset coordination section."""
        coord = data.get("coordination", {})
        lines = [
            "## 6. Cross-Asset Coordination\n",
            f"- **Coordination Score:** {self._fmt(coord.get('coordination_score', 0), 1)}/100",
            f"- **Dispatch Sequence:** {coord.get('dispatch_sequence', '-')}",
            f"- **Response Latency:** {coord.get('response_latency_seconds', 0)} seconds",
            f"- **Communication Protocol:** {coord.get('communication_protocol', '-')}",
            f"- **Overlapping Assets:** {coord.get('overlapping_assets', 0)}",
            f"- **Synergy Benefit:** {self._format_power(coord.get('synergy_benefit_kw', 0))}",
        ]
        return "\n".join(lines)

    def _md_recommendations(self, data: Dict[str, Any]) -> str:
        """Render DER optimization recommendations."""
        recs = data.get("recommendations", [])
        if not recs:
            recs = [
                "Increase battery pre-charge threshold before anticipated events",
                "Implement V2G capability for suitable EV fleet vehicles",
                "Add thermal storage pre-conditioning to dispatch plan",
                "Improve solar forecast integration for DER coordination",
            ]
        lines = ["## 7. DER Optimization Recommendations\n"]
        for i, rec in enumerate(recs, 1):
            lines.append(f"{i}. {rec}")
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render report footer."""
        return "---\n\n*Generated by GreenLang PACK-037 Demand Response Pack*"

    # ------------------------------------------------------------------
    # HTML section renderers
    # ------------------------------------------------------------------

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        facility = data.get("facility_name", "Facility")
        return (
            f'<h1>DER Performance Report</h1>\n'
            f'<p class="subtitle">Facility: {facility} | '
            f'Event: {data.get("event_id", "-")} | '
            f'DER Assets: {data.get("total_der_assets", 0)}</p>'
        )

    def _html_fleet_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML fleet summary cards."""
        s = data.get("fleet_summary", {})
        return (
            '<h2>DER Fleet Summary</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Total Assets</span>'
            f'<span class="value">{s.get("total_assets", 0)}</span></div>\n'
            f'  <div class="card"><span class="label">Installed kW</span>'
            f'<span class="value">{self._fmt(s.get("total_capacity_kw", 0), 0)}</span></div>\n'
            f'  <div class="card"><span class="label">Delivered kW</span>'
            f'<span class="value">{self._fmt(s.get("delivered_capacity_kw", 0), 0)}</span></div>\n'
            f'  <div class="card"><span class="label">Utilization</span>'
            f'<span class="value">{self._fmt(s.get("fleet_utilization_pct", 0))}%</span></div>\n'
            f'  <div class="card"><span class="label">Score</span>'
            f'<span class="value">{self._fmt(s.get("performance_score", 0), 1)}/100</span></div>\n'
            '</div>'
        )

    def _html_battery_performance(self, data: Dict[str, Any]) -> str:
        """Render HTML battery performance table."""
        batteries = data.get("battery_performance", [])
        rows = ""
        for b in batteries:
            rows += (
                f'<tr><td>{b.get("name", "-")}</td>'
                f'<td>{self._fmt(b.get("capacity_kwh", 0), 1)}</td>'
                f'<td>{self._fmt(b.get("start_soc_pct", 0), 1)}%</td>'
                f'<td>{self._fmt(b.get("end_soc_pct", 0), 1)}%</td>'
                f'<td>{self._fmt(b.get("delivered_kw", 0), 1)}</td></tr>\n'
            )
        return (
            '<h2>Battery Performance</h2>\n'
            '<table>\n<tr><th>Battery</th><th>Capacity kWh</th>'
            f'<th>Start SOC</th><th>End SOC</th><th>Delivered kW</th></tr>\n{rows}</table>'
        )

    def _html_solar_contribution(self, data: Dict[str, Any]) -> str:
        """Render HTML solar contribution."""
        solar = data.get("solar_contribution", {})
        return (
            '<h2>Solar PV Contribution</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Installed</span>'
            f'<span class="value">{self._fmt(solar.get("installed_capacity_kw", 0), 0)} kW</span></div>\n'
            f'  <div class="card"><span class="label">DR Contribution</span>'
            f'<span class="value">{self._fmt(solar.get("dr_contribution_kw", 0), 0)} kW</span></div>\n'
            f'  <div class="card"><span class="label">Avg Output</span>'
            f'<span class="value">{self._fmt(solar.get("avg_output_pct", 0))}%</span></div>\n'
            f'  <div class="card"><span class="label">Forecast Accuracy</span>'
            f'<span class="value">{self._fmt(solar.get("forecast_accuracy_pct", 0))}%</span></div>\n'
            '</div>'
        )

    def _html_ev_flexibility(self, data: Dict[str, Any]) -> str:
        """Render HTML EV flexibility summary."""
        ev = data.get("ev_flexibility", {})
        return (
            '<h2>EV Fleet Flexibility</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Chargers</span>'
            f'<span class="value">{ev.get("total_chargers", 0)}</span></div>\n'
            f'  <div class="card"><span class="label">Curtailed Load</span>'
            f'<span class="value">{self._fmt(ev.get("curtailed_load_kw", 0), 0)} kW</span></div>\n'
            f'  <div class="card"><span class="label">V2G Discharge</span>'
            f'<span class="value">{self._fmt(ev.get("v2g_discharge_kw", 0), 0)} kW</span></div>\n'
            f'  <div class="card"><span class="label">EVs Active</span>'
            f'<span class="value">{ev.get("evs_participating", 0)}</span></div>\n'
            '</div>'
        )

    def _html_thermal_storage(self, data: Dict[str, Any]) -> str:
        """Render HTML thermal storage summary."""
        thermal = data.get("thermal_storage", {})
        return (
            '<h2>Thermal Storage</h2>\n'
            f'<p>Total Capacity: {self._fmt(thermal.get("total_capacity_kwh_th", 0), 0)} kWh-th | '
            f'Load Deferred: {self._fmt(thermal.get("electrical_load_deferred_kw", 0), 0)} kW | '
            f'Pre-Conditioning: {thermal.get("pre_conditioning_hours", 0)} hrs</p>'
        )

    def _html_coordination(self, data: Dict[str, Any]) -> str:
        """Render HTML coordination metrics."""
        coord = data.get("coordination", {})
        return (
            '<h2>Cross-Asset Coordination</h2>\n'
            f'<div class="coordination-box">'
            f'<strong>Score: {self._fmt(coord.get("coordination_score", 0), 1)}/100</strong> | '
            f'Latency: {coord.get("response_latency_seconds", 0)}s | '
            f'Synergy: +{self._fmt(coord.get("synergy_benefit_kw", 0), 0)} kW</div>'
        )

    def _html_recommendations(self, data: Dict[str, Any]) -> str:
        """Render HTML recommendations."""
        recs = data.get("recommendations", [
            "Increase battery pre-charge threshold",
            "Implement V2G for suitable vehicles",
        ])
        items = "".join(f'<li>{r}</li>\n' for r in recs)
        return f'<h2>DER Optimization Recommendations</h2>\n<ol>\n{items}</ol>'

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_fleet_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON fleet summary."""
        s = data.get("fleet_summary", {})
        return {
            "total_assets": s.get("total_assets", 0),
            "total_capacity_kw": s.get("total_capacity_kw", 0),
            "dispatched_capacity_kw": s.get("dispatched_capacity_kw", 0),
            "delivered_capacity_kw": s.get("delivered_capacity_kw", 0),
            "fleet_utilization_pct": s.get("fleet_utilization_pct", 0),
            "performance_score": s.get("performance_score", 0),
            "battery_count": s.get("battery_count", 0),
            "solar_count": s.get("solar_count", 0),
            "ev_count": s.get("ev_count", 0),
            "thermal_count": s.get("thermal_count", 0),
        }

    def _json_charts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chart data payloads for visualization."""
        batteries = data.get("battery_performance", [])
        fleet = data.get("fleet_summary", {})
        return {
            "battery_soc_bar": {
                "type": "bar",
                "labels": [b.get("name", "") for b in batteries],
                "series": {
                    "start_soc": [b.get("start_soc_pct", 0) for b in batteries],
                    "end_soc": [b.get("end_soc_pct", 0) for b in batteries],
                },
            },
            "der_type_pie": {
                "type": "pie",
                "labels": ["Battery", "Solar", "EV", "Thermal"],
                "values": [
                    fleet.get("battery_count", 0),
                    fleet.get("solar_count", 0),
                    fleet.get("ev_count", 0),
                    fleet.get("thermal_count", 0),
                ],
            },
            "battery_delivered_bar": {
                "type": "bar",
                "labels": [b.get("name", "") for b in batteries],
                "values": [b.get("delivered_kw", 0) for b in batteries],
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
            ".coordination-box{background:#d1e7dd;padding:15px;border-radius:8px;margin:10px 0;}"
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
