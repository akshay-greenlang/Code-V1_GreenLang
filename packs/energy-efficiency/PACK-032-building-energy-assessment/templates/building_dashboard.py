# -*- coding: utf-8 -*-
"""
BuildingDashboardTemplate - Real-time building energy dashboard for PACK-032.

Generates real-time building energy dashboard reports with KPI summary
cards, energy consumption trends, end-use breakdowns, cost tracking,
carbon intensity metrics, weather normalization, alerts and anomalies,
target vs actuals tracking, and occupancy correlation analysis.

Sections:
    1. KPI Summary Cards
    2. Energy Consumption Trend
    3. End-Use Breakdown
    4. Cost Tracking
    5. Carbon Intensity
    6. Weather Normalization
    7. Alerts & Anomalies
    8. Targets vs Actuals
    9. Occupancy Correlation
   10. Provenance

Author: GreenLang Team
Version: 32.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class BuildingDashboardTemplate:
    """
    Real-time building energy dashboard template.

    Renders building energy dashboards with KPI cards, consumption
    trends, cost tracking, carbon metrics, weather normalization,
    alerts, and occupancy correlation data across markdown, HTML,
    and JSON formats.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    DASHBOARD_SECTIONS: List[str] = [
        "KPI Summary",
        "Consumption Trend",
        "End-Use Breakdown",
        "Cost Tracking",
        "Carbon Intensity",
        "Weather Normalization",
        "Alerts",
        "Targets vs Actuals",
        "Occupancy Correlation",
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize BuildingDashboardTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render building dashboard as Markdown.

        Args:
            data: Dashboard data from engine processing.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_kpi_summary(data),
            self._md_consumption_trend(data),
            self._md_end_use_breakdown(data),
            self._md_cost_tracking(data),
            self._md_carbon_intensity(data),
            self._md_weather_normalization(data),
            self._md_alerts(data),
            self._md_targets_vs_actuals(data),
            self._md_occupancy_correlation(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render building dashboard as self-contained HTML.

        Args:
            data: Dashboard data from engine processing.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_kpi_summary(data),
            self._html_consumption_trend(data),
            self._html_end_use_breakdown(data),
            self._html_cost_tracking(data),
            self._html_carbon_intensity(data),
            self._html_weather_normalization(data),
            self._html_alerts(data),
            self._html_targets_vs_actuals(data),
            self._html_occupancy_correlation(data),
        ])
        prov = self._provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Building Energy Dashboard</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render building dashboard as structured JSON.

        Args:
            data: Dashboard data from engine processing.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "building_dashboard",
            "version": "32.0.0",
            "generated_at": self.generated_at.isoformat(),
            "kpi_summary": data.get("kpi_summary", {}),
            "consumption_trend": data.get("consumption_trend", []),
            "end_use_breakdown": data.get("end_use_breakdown", []),
            "cost_tracking": data.get("cost_tracking", {}),
            "carbon_intensity": data.get("carbon_intensity", {}),
            "weather_normalization": data.get("weather_normalization", {}),
            "alerts": data.get("alerts", []),
            "targets_vs_actuals": data.get("targets_vs_actuals", {}),
            "occupancy_correlation": data.get("occupancy_correlation", {}),
            "charts": self._json_charts(data),
        }
        prov = self._provenance(json.dumps(result, default=str))
        result["provenance_hash"] = prov
        return result

    # ------------------------------------------------------------------
    # Markdown section renderers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header."""
        name = data.get("building_name", "Building")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            "# Building Energy Dashboard\n\n"
            f"**Building:** {name}  \n"
            f"**Dashboard Period:** {data.get('period', '-')}  \n"
            f"**Last Data Update:** {data.get('last_update', '-')}  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-032 BuildingDashboardTemplate v32.0.0\n\n---"
        )

    def _md_kpi_summary(self, data: Dict[str, Any]) -> str:
        """Render KPI summary cards section."""
        kpi = data.get("kpi_summary", {})
        return (
            "## 1. KPI Summary\n\n"
            f"| KPI | Value | Target | Status |\n"
            f"|-----|-------|--------|--------|\n"
            f"| EUI (kWh/m2) | {self._fmt(kpi.get('eui', 0))} "
            f"| {self._fmt(kpi.get('eui_target', 0))} "
            f"| {kpi.get('eui_status', '-')} |\n"
            f"| Energy Cost (MTD) | {kpi.get('cost_mtd', '-')} "
            f"| {kpi.get('cost_target', '-')} "
            f"| {kpi.get('cost_status', '-')} |\n"
            f"| Carbon (kgCO2/m2) | {self._fmt(kpi.get('carbon', 0))} "
            f"| {self._fmt(kpi.get('carbon_target', 0))} "
            f"| {kpi.get('carbon_status', '-')} |\n"
            f"| Active Alerts | {kpi.get('active_alerts', 0)} "
            f"| 0 | {kpi.get('alerts_status', '-')} |\n"
            f"| Occupancy | {self._fmt(kpi.get('occupancy_pct', 0))}% "
            f"| - | - |\n"
            f"| Renewable Share | {self._fmt(kpi.get('renewable_pct', 0))}% "
            f"| {self._fmt(kpi.get('renewable_target', 0))}% "
            f"| {kpi.get('renewable_status', '-')} |"
        )

    def _md_consumption_trend(self, data: Dict[str, Any]) -> str:
        """Render energy consumption trend section."""
        trend = data.get("consumption_trend", [])
        if not trend:
            return "## 2. Energy Consumption Trend\n\n_No trend data._"
        lines = [
            "## 2. Energy Consumption Trend\n",
            "| Period | Electricity (kWh) | Gas (kWh) | Total (kWh) | vs Target | vs Last Year |",
            "|--------|-------------------|-----------|-------------|-----------|-------------|",
        ]
        for t in trend:
            lines.append(
                f"| {t.get('period', '-')} "
                f"| {self._fmt(t.get('electricity_kwh', 0), 0)} "
                f"| {self._fmt(t.get('gas_kwh', 0), 0)} "
                f"| {self._fmt(t.get('total_kwh', 0), 0)} "
                f"| {t.get('vs_target', '-')} "
                f"| {t.get('vs_last_year', '-')} |"
            )
        return "\n".join(lines)

    def _md_end_use_breakdown(self, data: Dict[str, Any]) -> str:
        """Render end-use breakdown section."""
        end_uses = data.get("end_use_breakdown", [])
        if not end_uses:
            return "## 3. End-Use Breakdown\n\n_No end-use data._"
        lines = [
            "## 3. End-Use Breakdown\n",
            "| End Use | kWh | Share (%) | vs Benchmark | Trend |",
            "|---------|-----|-----------|-------------|-------|",
        ]
        for eu in end_uses:
            lines.append(
                f"| {eu.get('end_use', '-')} "
                f"| {self._fmt(eu.get('kwh', 0), 0)} "
                f"| {self._fmt(eu.get('share_pct', 0))}% "
                f"| {eu.get('vs_benchmark', '-')} "
                f"| {eu.get('trend', '-')} |"
            )
        return "\n".join(lines)

    def _md_cost_tracking(self, data: Dict[str, Any]) -> str:
        """Render cost tracking section."""
        cost = data.get("cost_tracking", {})
        monthly = cost.get("monthly", [])
        lines = [
            "## 4. Cost Tracking\n",
            f"**Year-to-Date Cost:** {cost.get('ytd_cost', '-')}  ",
            f"**Annual Budget:** {cost.get('annual_budget', '-')}  ",
            f"**Budget Utilization:** {self._fmt(cost.get('budget_utilization_pct', 0))}%  ",
            f"**Projected Annual:** {cost.get('projected_annual', '-')}  ",
            f"**vs Last Year:** {cost.get('vs_last_year', '-')}",
        ]
        if monthly:
            lines.extend([
                "\n### Monthly Cost Tracking\n",
                "| Month | Actual | Budget | Variance | Cumulative |",
                "|-------|--------|--------|----------|-----------|",
            ])
            for m in monthly:
                lines.append(
                    f"| {m.get('month', '-')} "
                    f"| {m.get('actual', '-')} "
                    f"| {m.get('budget', '-')} "
                    f"| {m.get('variance', '-')} "
                    f"| {m.get('cumulative', '-')} |"
                )
        return "\n".join(lines)

    def _md_carbon_intensity(self, data: Dict[str, Any]) -> str:
        """Render carbon intensity section."""
        ci = data.get("carbon_intensity", {})
        return (
            "## 5. Carbon Intensity\n\n"
            f"| Metric | Value |\n|--------|-------|\n"
            f"| Total CO2 (MTD) | {self._fmt(ci.get('mtd_co2_kg', 0), 0)} kg |\n"
            f"| CO2 Intensity | {self._fmt(ci.get('kg_co2_m2', 0))} kgCO2/m2 |\n"
            f"| Grid Carbon Factor | {self._fmt(ci.get('grid_factor', 0), 3)} kgCO2/kWh |\n"
            f"| Scope 1 | {self._fmt(ci.get('scope1_kg', 0), 0)} kg |\n"
            f"| Scope 2 (Location) | {self._fmt(ci.get('scope2_location_kg', 0), 0)} kg |\n"
            f"| Scope 2 (Market) | {self._fmt(ci.get('scope2_market_kg', 0), 0)} kg |\n"
            f"| vs CRREM Target | {ci.get('vs_crrem', '-')} |\n"
            f"| Trend | {ci.get('trend', '-')} |"
        )

    def _md_weather_normalization(self, data: Dict[str, Any]) -> str:
        """Render weather normalization section."""
        wn = data.get("weather_normalization", {})
        return (
            "## 6. Weather Normalization\n\n"
            f"**Actual EUI:** {self._fmt(wn.get('actual_eui', 0))} kWh/m2  \n"
            f"**Normalized EUI:** {self._fmt(wn.get('normalized_eui', 0))} kWh/m2  \n"
            f"**Weather Impact:** {self._fmt(wn.get('weather_impact_pct', 0))}%  \n"
            f"**HDD (Actual/Normal):** {self._fmt(wn.get('actual_hdd', 0), 0)} / "
            f"{self._fmt(wn.get('normal_hdd', 0), 0)}  \n"
            f"**CDD (Actual/Normal):** {self._fmt(wn.get('actual_cdd', 0), 0)} / "
            f"{self._fmt(wn.get('normal_cdd', 0), 0)}"
        )

    def _md_alerts(self, data: Dict[str, Any]) -> str:
        """Render alerts and anomalies section."""
        alerts = data.get("alerts", [])
        if not alerts:
            return "## 7. Alerts & Anomalies\n\n_No active alerts._"
        lines = [
            "## 7. Alerts & Anomalies\n",
            "| # | Severity | Type | Description | Detected | Status |",
            "|---|----------|------|-------------|----------|--------|",
        ]
        for i, a in enumerate(alerts, 1):
            lines.append(
                f"| {i} | {a.get('severity', '-')} "
                f"| {a.get('type', '-')} "
                f"| {a.get('description', '-')} "
                f"| {a.get('detected', '-')} "
                f"| {a.get('status', '-')} |"
            )
        return "\n".join(lines)

    def _md_targets_vs_actuals(self, data: Dict[str, Any]) -> str:
        """Render targets vs actuals section."""
        tva = data.get("targets_vs_actuals", {})
        metrics = tva.get("metrics", [])
        if not metrics:
            return "## 8. Targets vs Actuals\n\n_No target data._"
        lines = [
            "## 8. Targets vs Actuals\n",
            "| Metric | Target | Actual | Variance | On Track |",
            "|--------|--------|--------|----------|----------|",
        ]
        for m in metrics:
            lines.append(
                f"| {m.get('metric', '-')} "
                f"| {m.get('target', '-')} "
                f"| {m.get('actual', '-')} "
                f"| {m.get('variance', '-')} "
                f"| {m.get('on_track', '-')} |"
            )
        return "\n".join(lines)

    def _md_occupancy_correlation(self, data: Dict[str, Any]) -> str:
        """Render occupancy correlation section."""
        occ = data.get("occupancy_correlation", {})
        data_points = occ.get("data_points", [])
        lines = [
            "## 9. Occupancy Correlation\n",
            f"**Correlation Coefficient:** {self._fmt(occ.get('correlation', 0), 3)}  ",
            f"**Base Load (kWh):** {self._fmt(occ.get('base_load_kwh', 0), 0)}  ",
            f"**Per Person Load (kWh):** {self._fmt(occ.get('per_person_kwh', 0))}  ",
            f"**R-Squared:** {self._fmt(occ.get('r_squared', 0), 3)}",
        ]
        if data_points:
            lines.extend([
                "\n### Occupancy vs Energy Data\n",
                "| Period | Occupancy (%) | Energy (kWh) | Expected (kWh) | Residual |",
                "|--------|--------------|-------------|----------------|----------|",
            ])
            for dp in data_points:
                lines.append(
                    f"| {dp.get('period', '-')} "
                    f"| {self._fmt(dp.get('occupancy_pct', 0))}% "
                    f"| {self._fmt(dp.get('energy_kwh', 0), 0)} "
                    f"| {self._fmt(dp.get('expected_kwh', 0), 0)} "
                    f"| {self._fmt(dp.get('residual_kwh', 0), 0)} |"
                )
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render markdown footer."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            "---\n\n"
            f"*Dashboard generated by PACK-032 BuildingDashboardTemplate v32.0.0 on {ts}*"
        )

    # ------------------------------------------------------------------
    # HTML section renderers
    # ------------------------------------------------------------------

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        name = data.get("building_name", "Building")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<h1>Building Energy Dashboard</h1>\n'
            f'<p class="subtitle">Building: {name} | Period: {data.get("period", "-")} | '
            f'Generated: {ts}</p>'
        )

    def _html_kpi_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML KPI summary cards."""
        kpi = data.get("kpi_summary", {})
        return (
            '<h2>KPI Summary</h2>\n'
            '<div class="summary-cards">\n'
            f'<div class="card"><span class="label">EUI</span>'
            f'<span class="value">{self._fmt(kpi.get("eui", 0))}</span>'
            f'<span class="label">kWh/m2 ({kpi.get("eui_status", "-")})</span></div>\n'
            f'<div class="card"><span class="label">Cost MTD</span>'
            f'<span class="value">{kpi.get("cost_mtd", "-")}</span>'
            f'<span class="label">{kpi.get("cost_status", "-")}</span></div>\n'
            f'<div class="card"><span class="label">Carbon</span>'
            f'<span class="value">{self._fmt(kpi.get("carbon", 0))}</span>'
            f'<span class="label">kgCO2/m2</span></div>\n'
            f'<div class="card"><span class="label">Alerts</span>'
            f'<span class="value">{kpi.get("active_alerts", 0)}</span>'
            f'<span class="label">active</span></div>\n'
            f'<div class="card"><span class="label">Occupancy</span>'
            f'<span class="value">{self._fmt(kpi.get("occupancy_pct", 0))}%</span></div>\n'
            '</div>'
        )

    def _html_consumption_trend(self, data: Dict[str, Any]) -> str:
        """Render HTML consumption trend table."""
        trend = data.get("consumption_trend", [])
        rows = ""
        for t in trend:
            rows += (
                f'<tr><td>{t.get("period", "-")}</td>'
                f'<td>{self._fmt(t.get("electricity_kwh", 0), 0)}</td>'
                f'<td>{self._fmt(t.get("gas_kwh", 0), 0)}</td>'
                f'<td>{self._fmt(t.get("total_kwh", 0), 0)}</td>'
                f'<td>{t.get("vs_target", "-")}</td></tr>\n'
            )
        return (
            '<h2>Energy Consumption Trend</h2>\n'
            '<table>\n<tr><th>Period</th><th>Elec (kWh)</th><th>Gas (kWh)</th>'
            f'<th>Total</th><th>vs Target</th></tr>\n{rows}</table>'
        )

    def _html_end_use_breakdown(self, data: Dict[str, Any]) -> str:
        """Render HTML end-use breakdown."""
        end_uses = data.get("end_use_breakdown", [])
        rows = ""
        for eu in end_uses:
            rows += (
                f'<tr><td>{eu.get("end_use", "-")}</td>'
                f'<td>{self._fmt(eu.get("kwh", 0), 0)}</td>'
                f'<td>{self._fmt(eu.get("share_pct", 0))}%</td>'
                f'<td>{eu.get("trend", "-")}</td></tr>\n'
            )
        return (
            '<h2>End-Use Breakdown</h2>\n'
            '<table>\n<tr><th>End Use</th><th>kWh</th><th>Share</th>'
            f'<th>Trend</th></tr>\n{rows}</table>'
        )

    def _html_cost_tracking(self, data: Dict[str, Any]) -> str:
        """Render HTML cost tracking."""
        cost = data.get("cost_tracking", {})
        return (
            '<h2>Cost Tracking</h2>\n'
            f'<p>YTD: {cost.get("ytd_cost", "-")} | '
            f'Budget: {cost.get("annual_budget", "-")} | '
            f'Utilization: {self._fmt(cost.get("budget_utilization_pct", 0))}%</p>'
        )

    def _html_carbon_intensity(self, data: Dict[str, Any]) -> str:
        """Render HTML carbon intensity."""
        ci = data.get("carbon_intensity", {})
        return (
            '<h2>Carbon Intensity</h2>\n'
            '<div class="summary-cards">\n'
            f'<div class="card"><span class="label">CO2 MTD</span>'
            f'<span class="value">{self._fmt(ci.get("mtd_co2_kg", 0), 0)}</span>'
            f'<span class="label">kg</span></div>\n'
            f'<div class="card"><span class="label">Intensity</span>'
            f'<span class="value">{self._fmt(ci.get("kg_co2_m2", 0))}</span>'
            f'<span class="label">kgCO2/m2</span></div>\n'
            f'<div class="card"><span class="label">vs CRREM</span>'
            f'<span class="value">{ci.get("vs_crrem", "-")}</span></div>\n'
            '</div>'
        )

    def _html_weather_normalization(self, data: Dict[str, Any]) -> str:
        """Render HTML weather normalization."""
        wn = data.get("weather_normalization", {})
        return (
            '<h2>Weather Normalization</h2>\n'
            f'<p>Actual EUI: {self._fmt(wn.get("actual_eui", 0))} | '
            f'Normalized: {self._fmt(wn.get("normalized_eui", 0))} | '
            f'Impact: {self._fmt(wn.get("weather_impact_pct", 0))}%</p>'
        )

    def _html_alerts(self, data: Dict[str, Any]) -> str:
        """Render HTML alerts."""
        alerts = data.get("alerts", [])
        items = ""
        for a in alerts:
            severity = a.get("severity", "info")
            bg = "#fff3cd" if severity == "warning" else "#f8d7da" if severity == "critical" else "#d1ecf1"
            items += (
                f'<div class="alert-item" style="background:{bg};border-left:4px solid '
                f'{"#ffc107" if severity == "warning" else "#dc3545" if severity == "critical" else "#0dcaf0"};'
                f'padding:10px;margin:5px 0;">'
                f'<strong>[{severity.upper()}]</strong> {a.get("description", "-")} '
                f'<em>({a.get("detected", "-")})</em></div>\n'
            )
        return f'<h2>Alerts &amp; Anomalies</h2>\n{items}'

    def _html_targets_vs_actuals(self, data: Dict[str, Any]) -> str:
        """Render HTML targets vs actuals."""
        tva = data.get("targets_vs_actuals", {})
        metrics = tva.get("metrics", [])
        rows = ""
        for m in metrics:
            rows += (
                f'<tr><td>{m.get("metric", "-")}</td>'
                f'<td>{m.get("target", "-")}</td>'
                f'<td>{m.get("actual", "-")}</td>'
                f'<td>{m.get("on_track", "-")}</td></tr>\n'
            )
        return (
            '<h2>Targets vs Actuals</h2>\n'
            '<table>\n<tr><th>Metric</th><th>Target</th><th>Actual</th>'
            f'<th>On Track</th></tr>\n{rows}</table>'
        )

    def _html_occupancy_correlation(self, data: Dict[str, Any]) -> str:
        """Render HTML occupancy correlation."""
        occ = data.get("occupancy_correlation", {})
        return (
            '<h2>Occupancy Correlation</h2>\n'
            f'<p>Correlation: {self._fmt(occ.get("correlation", 0), 3)} | '
            f'R2: {self._fmt(occ.get("r_squared", 0), 3)} | '
            f'Base Load: {self._fmt(occ.get("base_load_kwh", 0), 0)} kWh</p>'
        )

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_charts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chart data payloads for visualization."""
        trend = data.get("consumption_trend", [])
        end_uses = data.get("end_use_breakdown", [])
        occ = data.get("occupancy_correlation", {}).get("data_points", [])
        return {
            "consumption_line": {
                "type": "line",
                "labels": [t.get("period", "") for t in trend],
                "series": {
                    "electricity": [t.get("electricity_kwh", 0) for t in trend],
                    "gas": [t.get("gas_kwh", 0) for t in trend],
                    "total": [t.get("total_kwh", 0) for t in trend],
                },
            },
            "end_use_pie": {
                "type": "pie",
                "labels": [eu.get("end_use", "") for eu in end_uses],
                "values": [eu.get("kwh", 0) for eu in end_uses],
            },
            "occupancy_scatter": {
                "type": "scatter",
                "x": [dp.get("occupancy_pct", 0) for dp in occ],
                "y": [dp.get("energy_kwh", 0) for dp in occ],
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
            ".card{background:#f8f9fa;border-radius:8px;padding:15px;flex:1;text-align:center;min-width:140px;}"
            ".label{display:block;font-size:0.85em;color:#6c757d;}"
            ".value{display:block;font-size:1.4em;font-weight:700;color:#198754;}"
            ".subtitle{color:#6c757d;font-size:0.95em;}"
        )

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

    def _provenance(self, content: str) -> str:
        """Compute SHA-256 provenance hash.

        Args:
            content: Content string to hash.

        Returns:
            SHA-256 hex digest.
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
