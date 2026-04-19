# -*- coding: utf-8 -*-
"""
EnergyManagementDashboardTemplate - Real-time energy dashboard for PACK-031.

Generates structured dashboard data for energy management with KPIs,
trend charts, active alerts, EnPI tracking, target progress monitoring,
cost tracking, and weather-normalized consumption data. Designed for
integration with Grafana or web-based dashboard rendering.

Sections:
    1. KPI Summary Cards
    2. Consumption Trend Charts
    3. Active Alerts & Notifications
    4. EnPI Tracking
    5. Target Progress
    6. Cost & Budget Tracking
    7. Weather-Normalized Consumption
    8. System Status

Author: GreenLang Team
Version: 31.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class EnergyManagementDashboardTemplate:
    """
    Real-time energy management dashboard template.

    Renders dashboard data with KPIs, trend charts, alerts, EnPI tracking,
    target progress, and cost monitoring across markdown, HTML, and JSON.
    Optimized for periodic refresh and real-time data feeds.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    ALERT_SEVERITIES: List[str] = ["critical", "warning", "info"]

    KPI_CATEGORIES: List[str] = [
        "total_consumption",
        "specific_energy",
        "peak_demand",
        "power_factor",
        "cost_per_unit",
        "renewable_share",
        "savings_ytd",
        "target_progress",
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize EnergyManagementDashboardTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render energy management dashboard as Markdown.

        Args:
            data: Dashboard data from engine processing.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_kpi_summary(data),
            self._md_consumption_trends(data),
            self._md_active_alerts(data),
            self._md_enpi_tracking(data),
            self._md_target_progress(data),
            self._md_cost_tracking(data),
            self._md_weather_normalized(data),
            self._md_system_status(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render energy management dashboard as self-contained HTML.

        Args:
            data: Dashboard data from engine processing.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_kpi_cards(data),
            self._html_alerts(data),
            self._html_enpi_tracking(data),
            self._html_target_progress(data),
            self._html_cost_tracking(data),
            self._html_weather_normalized(data),
        ])
        prov = self._provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Energy Management Dashboard</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="dashboard">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render energy management dashboard as structured JSON.

        Args:
            data: Dashboard data from engine processing.

        Returns:
            Dict with structured dashboard sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "energy_management_dashboard",
            "version": "31.0.0",
            "generated_at": self.generated_at.isoformat(),
            "kpis": self._json_kpis(data),
            "consumption_trends": data.get("consumption_trends", {}),
            "alerts": data.get("alerts", []),
            "enpi_tracking": data.get("enpi_tracking", {}),
            "target_progress": data.get("target_progress", {}),
            "cost_tracking": data.get("cost_tracking", {}),
            "weather_normalized": data.get("weather_normalized", {}),
            "system_status": data.get("system_status", {}),
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
        facility = data.get("facility_name", "Industrial Facility")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        period = data.get("reporting_period", "Current Month")
        return (
            f"# Energy Management Dashboard\n\n"
            f"**Facility:** {facility}  \n"
            f"**Period:** {period}  \n"
            f"**Last Updated:** {ts}  \n"
            f"**Template:** PACK-031 EnergyManagementDashboardTemplate v31.0.0\n\n---"
        )

    def _md_kpi_summary(self, data: Dict[str, Any]) -> str:
        """Render KPI summary cards section."""
        kpis = data.get("kpis", {})
        lines = [
            "## Key Performance Indicators\n",
            "| KPI | Current | Target | Trend | Status |",
            "|-----|---------|--------|-------|--------|",
        ]
        for kpi in kpis.get("indicators", []):
            trend_arrow = self._trend_arrow(kpi.get("trend", "flat"))
            lines.append(
                f"| {kpi.get('name', '-')} "
                f"| {self._fmt(kpi.get('current_value', 0))} {kpi.get('unit', '')} "
                f"| {self._fmt(kpi.get('target_value', 0))} {kpi.get('unit', '')} "
                f"| {trend_arrow} {self._fmt(kpi.get('change_pct', 0))}% "
                f"| {kpi.get('status', '-')} |"
            )
        return "\n".join(lines)

    def _md_consumption_trends(self, data: Dict[str, Any]) -> str:
        """Render consumption trends section."""
        trends = data.get("consumption_trends", {})
        daily = trends.get("daily", [])
        lines = [
            "## Consumption Trends\n",
            f"**Period Total:** {self._fmt(trends.get('period_total_mwh', 0))} MWh  ",
            f"**Daily Average:** {self._fmt(trends.get('daily_average_mwh', 0))} MWh  ",
            f"**Peak Day:** {trends.get('peak_day', '-')} "
            f"({self._fmt(trends.get('peak_day_mwh', 0))} MWh)  ",
            f"**Minimum Day:** {trends.get('min_day', '-')} "
            f"({self._fmt(trends.get('min_day_mwh', 0))} MWh)",
        ]
        if daily:
            lines.extend([
                "\n### Daily Consumption (Last 7 Days)\n",
                "| Date | Electricity (MWh) | Gas (MWh) | Total (MWh) | vs. Baseline |",
                "|------|------------------|-----------|-------------|-------------|",
            ])
            for d in daily[-7:]:
                lines.append(
                    f"| {d.get('date', '-')} "
                    f"| {self._fmt(d.get('electricity_mwh', 0))} "
                    f"| {self._fmt(d.get('gas_mwh', 0))} "
                    f"| {self._fmt(d.get('total_mwh', 0))} "
                    f"| {self._fmt(d.get('vs_baseline_pct', 0))}% |"
                )
        return "\n".join(lines)

    def _md_active_alerts(self, data: Dict[str, Any]) -> str:
        """Render active alerts section."""
        alerts = data.get("alerts", [])
        if not alerts:
            return "## Active Alerts\n\n_No active alerts._"
        lines = [
            "## Active Alerts\n",
            "| Severity | Alert | Area | Triggered | Value | Threshold |",
            "|----------|-------|------|-----------|-------|-----------|",
        ]
        for a in alerts:
            lines.append(
                f"| {a.get('severity', '-').upper()} "
                f"| {a.get('message', '-')} "
                f"| {a.get('area', '-')} "
                f"| {a.get('triggered_at', '-')} "
                f"| {self._fmt(a.get('current_value', 0))} "
                f"| {self._fmt(a.get('threshold', 0))} |"
            )
        return "\n".join(lines)

    def _md_enpi_tracking(self, data: Dict[str, Any]) -> str:
        """Render EnPI tracking section."""
        enpi = data.get("enpi_tracking", {})
        indicators = enpi.get("indicators", [])
        lines = [
            "## EnPI Tracking\n",
            f"**Primary EnPI:** {self._fmt(enpi.get('primary_enpi_value', 0))} "
            f"{enpi.get('primary_enpi_unit', 'kWh/unit')}  ",
            f"**Baseline EnPI:** {self._fmt(enpi.get('baseline_enpi', 0))} "
            f"{enpi.get('primary_enpi_unit', 'kWh/unit')}  ",
            f"**Improvement:** {self._fmt(enpi.get('improvement_pct', 0))}%",
        ]
        if indicators:
            lines.extend([
                "\n| Period | EnPI | Baseline | Variance | Cumulative Savings |",
                "|--------|------|----------|----------|-------------------|",
            ])
            for ind in indicators[-12:]:
                lines.append(
                    f"| {ind.get('period', '-')} "
                    f"| {self._fmt(ind.get('value', 0))} "
                    f"| {self._fmt(ind.get('baseline', 0))} "
                    f"| {self._fmt(ind.get('variance_pct', 0))}% "
                    f"| {self._fmt(ind.get('cumulative_savings', 0))} |"
                )
        return "\n".join(lines)

    def _md_target_progress(self, data: Dict[str, Any]) -> str:
        """Render target progress section."""
        targets = data.get("target_progress", {})
        items = targets.get("targets", [])
        lines = [
            "## Target Progress\n",
            f"**Overall Progress:** {self._fmt(targets.get('overall_progress_pct', 0))}%  ",
            f"**On Track:** {targets.get('on_track_count', 0)} of "
            f"{targets.get('total_targets', 0)} targets",
        ]
        if items:
            lines.extend([
                "\n| Target | Baseline | Current | Goal | Progress | Status |",
                "|--------|----------|---------|------|----------|--------|",
            ])
            for t in items:
                progress_bar = self._progress_bar(t.get("progress_pct", 0))
                lines.append(
                    f"| {t.get('name', '-')} "
                    f"| {self._fmt(t.get('baseline_value', 0))} "
                    f"| {self._fmt(t.get('current_value', 0))} "
                    f"| {self._fmt(t.get('target_value', 0))} "
                    f"| {progress_bar} {self._fmt(t.get('progress_pct', 0))}% "
                    f"| {t.get('status', '-')} |"
                )
        return "\n".join(lines)

    def _md_cost_tracking(self, data: Dict[str, Any]) -> str:
        """Render cost and budget tracking section."""
        cost = data.get("cost_tracking", {})
        lines = [
            "## Cost & Budget Tracking\n",
            "| Metric | Value |",
            "|--------|-------|",
            f"| YTD Energy Cost | EUR {self._fmt(cost.get('ytd_cost_eur', 0))} |",
            f"| YTD Budget | EUR {self._fmt(cost.get('ytd_budget_eur', 0))} |",
            f"| Budget Variance | EUR {self._fmt(cost.get('budget_variance_eur', 0))} "
            f"({self._fmt(cost.get('budget_variance_pct', 0))}%) |",
            f"| Avg Unit Cost (Electricity) | EUR {self._fmt(cost.get('avg_elec_cost_eur_mwh', 0))}/MWh |",
            f"| Avg Unit Cost (Gas) | EUR {self._fmt(cost.get('avg_gas_cost_eur_mwh', 0))}/MWh |",
            f"| Projected Annual Cost | EUR {self._fmt(cost.get('projected_annual_eur', 0))} |",
            f"| Cost Savings YTD | EUR {self._fmt(cost.get('savings_ytd_eur', 0))} |",
        ]
        return "\n".join(lines)

    def _md_weather_normalized(self, data: Dict[str, Any]) -> str:
        """Render weather-normalized consumption section."""
        wn = data.get("weather_normalized", {})
        monthly = wn.get("monthly", [])
        lines = [
            "## Weather-Normalized Consumption\n",
            f"**Normalization Method:** {wn.get('method', 'Degree-Day Regression')}  ",
            f"**HDD Base:** {wn.get('hdd_base_c', 15.5)} C  ",
            f"**CDD Base:** {wn.get('cdd_base_c', 18.0)} C",
        ]
        if monthly:
            lines.extend([
                "\n| Month | Actual (MWh) | Normalized (MWh) | Adjustment (%) |",
                "|-------|-------------|------------------|---------------|",
            ])
            for m in monthly[-12:]:
                lines.append(
                    f"| {m.get('month', '-')} "
                    f"| {self._fmt(m.get('actual_mwh', 0))} "
                    f"| {self._fmt(m.get('normalized_mwh', 0))} "
                    f"| {self._fmt(m.get('adjustment_pct', 0))}% |"
                )
        return "\n".join(lines)

    def _md_system_status(self, data: Dict[str, Any]) -> str:
        """Render system status section."""
        status = data.get("system_status", {})
        meters = status.get("meters", [])
        lines = [
            "## System Status\n",
            f"**Active Meters:** {status.get('active_meters', 0)} "
            f"of {status.get('total_meters', 0)}  ",
            f"**Data Completeness:** {self._fmt(status.get('data_completeness_pct', 0))}%  ",
            f"**Last Data Sync:** {status.get('last_sync', '-')}",
        ]
        if meters:
            lines.extend([
                "\n| Meter | Location | Status | Last Reading | Data Quality |",
                "|-------|----------|--------|-------------|-------------|",
            ])
            for m in meters:
                lines.append(
                    f"| {m.get('meter_id', '-')} "
                    f"| {m.get('location', '-')} "
                    f"| {m.get('status', '-')} "
                    f"| {m.get('last_reading', '-')} "
                    f"| {m.get('data_quality', '-')} |"
                )
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render dashboard footer."""
        return "---\n\n*Generated by GreenLang PACK-031 Industrial Energy Audit Pack*"

    # ------------------------------------------------------------------
    # HTML section renderers
    # ------------------------------------------------------------------

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        facility = data.get("facility_name", "Industrial Facility")
        ts = self.generated_at.strftime("%H:%M UTC") if self.generated_at else ""
        return (
            f'<div class="dash-header"><h1>Energy Management Dashboard</h1>'
            f'<p>{facility} | Updated: {ts}</p></div>'
        )

    def _html_kpi_cards(self, data: Dict[str, Any]) -> str:
        """Render HTML KPI cards."""
        kpis = data.get("kpis", {}).get("indicators", [])
        cards = ""
        for kpi in kpis[:8]:
            color = "#059669" if kpi.get("status") == "on_track" else "#d97706"
            cards += (
                f'<div class="kpi-card">'
                f'<span class="kpi-label">{kpi.get("name", "-")}</span>'
                f'<span class="kpi-value" style="color:{color};">'
                f'{self._fmt(kpi.get("current_value", 0))} {kpi.get("unit", "")}</span>'
                f'<span class="kpi-trend">{self._fmt(kpi.get("change_pct", 0))}%</span>'
                f'</div>\n'
            )
        return f'<h2>Key Performance Indicators</h2>\n<div class="kpi-grid">\n{cards}</div>'

    def _html_alerts(self, data: Dict[str, Any]) -> str:
        """Render HTML alerts."""
        alerts = data.get("alerts", [])
        if not alerts:
            return '<h2>Alerts</h2>\n<p class="no-alerts">No active alerts</p>'
        items = ""
        for a in alerts:
            sev = a.get("severity", "info")
            cls = f"alert-{sev}"
            items += (
                f'<div class="alert {cls}">'
                f'<strong>[{sev.upper()}]</strong> {a.get("message", "-")} '
                f'({a.get("area", "-")})</div>\n'
            )
        return f'<h2>Active Alerts</h2>\n{items}'

    def _html_enpi_tracking(self, data: Dict[str, Any]) -> str:
        """Render HTML EnPI tracking."""
        enpi = data.get("enpi_tracking", {})
        return (
            '<h2>EnPI Tracking</h2>\n'
            f'<p>Current: {self._fmt(enpi.get("primary_enpi_value", 0))} '
            f'{enpi.get("primary_enpi_unit", "kWh/unit")} | '
            f'Improvement: {self._fmt(enpi.get("improvement_pct", 0))}%</p>\n'
            '<div class="chart-placeholder" data-chart="enpi_trend">[EnPI Chart]</div>'
        )

    def _html_target_progress(self, data: Dict[str, Any]) -> str:
        """Render HTML target progress bars."""
        targets = data.get("target_progress", {}).get("targets", [])
        bars = ""
        for t in targets:
            pct = min(t.get("progress_pct", 0), 100)
            color = "#059669" if t.get("status") == "on_track" else "#d97706"
            bars += (
                f'<div class="target-row">'
                f'<span class="target-name">{t.get("name", "-")}</span>'
                f'<div class="progress-bar"><div class="progress-fill" '
                f'style="width:{pct}%;background:{color};"></div></div>'
                f'<span class="target-pct">{self._fmt(pct)}%</span></div>\n'
            )
        return f'<h2>Target Progress</h2>\n{bars}'

    def _html_cost_tracking(self, data: Dict[str, Any]) -> str:
        """Render HTML cost tracking."""
        cost = data.get("cost_tracking", {})
        return (
            '<h2>Cost Tracking</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">YTD Cost</span>'
            f'<span class="value">EUR {self._fmt(cost.get("ytd_cost_eur", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Budget Variance</span>'
            f'<span class="value">{self._fmt(cost.get("budget_variance_pct", 0))}%</span></div>\n'
            f'  <div class="card"><span class="label">Savings YTD</span>'
            f'<span class="value">EUR {self._fmt(cost.get("savings_ytd_eur", 0))}</span></div>\n'
            '</div>'
        )

    def _html_weather_normalized(self, data: Dict[str, Any]) -> str:
        """Render HTML weather-normalized section."""
        return (
            '<h2>Weather-Normalized Consumption</h2>\n'
            '<div class="chart-placeholder" data-chart="weather_normalized">'
            '[Weather-Normalized Chart]</div>'
        )

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_kpis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON KPI data."""
        return data.get("kpis", {})

    def _json_charts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chart data payloads for visualization."""
        trends = data.get("consumption_trends", {}).get("daily", [])
        enpi = data.get("enpi_tracking", {}).get("indicators", [])
        wn = data.get("weather_normalized", {}).get("monthly", [])
        return {
            "consumption_trend": {
                "type": "line",
                "labels": [d.get("date", "") for d in trends],
                "series": {
                    "electricity": [d.get("electricity_mwh", 0) for d in trends],
                    "gas": [d.get("gas_mwh", 0) for d in trends],
                    "total": [d.get("total_mwh", 0) for d in trends],
                },
            },
            "enpi_trend": {
                "type": "line",
                "labels": [e.get("period", "") for e in enpi],
                "series": {
                    "actual": [e.get("value", 0) for e in enpi],
                    "baseline": [e.get("baseline", 0) for e in enpi],
                },
            },
            "weather_comparison": {
                "type": "grouped_bar",
                "labels": [m.get("month", "") for m in wn],
                "series": {
                    "actual": [m.get("actual_mwh", 0) for m in wn],
                    "normalized": [m.get("normalized_mwh", 0) for m in wn],
                },
            },
            "cost_waterfall": {
                "type": "waterfall",
                "labels": ["Budget", "Electricity", "Gas", "Other", "Savings", "Actual"],
                "values": [
                    data.get("cost_tracking", {}).get("ytd_budget_eur", 0),
                    data.get("cost_tracking", {}).get("electricity_cost_eur", 0),
                    data.get("cost_tracking", {}).get("gas_cost_eur", 0),
                    data.get("cost_tracking", {}).get("other_cost_eur", 0),
                    -data.get("cost_tracking", {}).get("savings_ytd_eur", 0),
                    data.get("cost_tracking", {}).get("ytd_cost_eur", 0),
                ],
            },
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _trend_arrow(self, trend: str) -> str:
        """Return a text-based trend indicator.

        Args:
            trend: Trend direction ('up', 'down', 'flat').

        Returns:
            Arrow string indicator.
        """
        arrows = {"up": "[UP]", "down": "[DOWN]", "flat": "[--]"}
        return arrows.get(trend, "[--]")

    def _progress_bar(self, pct: float, width: int = 20) -> str:
        """Generate a text-based progress bar.

        Args:
            pct: Progress percentage (0-100).
            width: Character width of the bar.

        Returns:
            Text progress bar string.
        """
        filled = min(int(pct / 100 * width), width)
        return "[" + "#" * filled + "-" * (width - filled) + "]"

    def _css(self) -> str:
        """Build inline CSS for HTML rendering."""
        return (
            "body{font-family:system-ui,-apple-system,sans-serif;margin:0;padding:20px;"
            "background:#f0f2f5;color:#1a1a2e;}"
            ".dashboard{max-width:1400px;margin:0 auto;}"
            ".dash-header{background:#1a1a2e;color:#fff;padding:20px;border-radius:8px;margin-bottom:20px;}"
            ".dash-header h1{margin:0;color:#fff;}"
            ".dash-header p{margin:5px 0 0;opacity:0.8;}"
            "h2{color:#198754;margin-top:25px;}"
            ".kpi-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:15px 0;}"
            ".kpi-card{background:#fff;border-radius:8px;padding:15px;text-align:center;"
            "box-shadow:0 1px 3px rgba(0,0,0,0.1);}"
            ".kpi-label{display:block;font-size:0.8em;color:#6c757d;}"
            ".kpi-value{display:block;font-size:1.6em;font-weight:700;}"
            ".kpi-trend{display:block;font-size:0.85em;color:#6c757d;}"
            ".summary-cards{display:flex;gap:15px;margin:15px 0;}"
            ".card{background:#fff;border-radius:8px;padding:15px;flex:1;text-align:center;"
            "box-shadow:0 1px 3px rgba(0,0,0,0.1);}"
            ".label{display:block;font-size:0.85em;color:#6c757d;}"
            ".value{display:block;font-size:1.4em;font-weight:700;color:#198754;}"
            ".alert{padding:10px;border-radius:6px;margin:5px 0;}"
            ".alert-critical{background:#fee2e2;border-left:4px solid #dc2626;}"
            ".alert-warning{background:#fff3cd;border-left:4px solid #ffc107;}"
            ".alert-info{background:#d1ecf1;border-left:4px solid #0dcaf0;}"
            ".no-alerts{color:#059669;font-style:italic;}"
            ".target-row{display:flex;align-items:center;gap:10px;margin:8px 0;}"
            ".target-name{width:200px;font-weight:600;}"
            ".progress-bar{flex:1;height:20px;background:#e9ecef;border-radius:4px;overflow:hidden;}"
            ".progress-fill{height:100%;border-radius:4px;transition:width 0.3s;}"
            ".target-pct{width:60px;text-align:right;font-weight:600;}"
            ".chart-placeholder{background:#fff;border:2px dashed #dee2e6;padding:40px;"
            "text-align:center;margin:15px 0;border-radius:8px;}"
            "table{width:100%;border-collapse:collapse;margin:15px 0;background:#fff;}"
            "th,td{border:1px solid #dee2e6;padding:8px 12px;text-align:left;}"
            "th{background:#f8f9fa;font-weight:600;}"
        )

    def _fmt(self, val: Any, decimals: int = 2) -> str:
        """Format a numeric value with comma separators."""
        if isinstance(val, (int, float)):
            return f"{val:,.{decimals}f}"
        return str(val)

    def _provenance(self, content: str) -> str:
        """Compute SHA-256 provenance hash."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
