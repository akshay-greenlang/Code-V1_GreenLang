# -*- coding: utf-8 -*-
"""
ProgressDashboardTemplate - Progress tracking dashboard for PACK-033.

Generates progress tracking dashboards for quick-win energy efficiency
measure implementations, including KPI cards, savings trends, implementation
status by measure, variance analysis, and alerts.

Sections:
    1. KPI Cards (total savings, ROI, completion rate, CO2e)
    2. Savings Trend (monthly)
    3. Implementation Status (by measure)
    4. Variance Analysis
    5. Alerts

Author: GreenLang Team
Version: 33.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ProgressDashboardTemplate:
    """
    Progress tracking dashboard template.

    Renders implementation progress dashboards with KPI cards,
    savings trend charts, status tracking, variance analysis, and
    alerts across markdown, HTML, and JSON formats.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize ProgressDashboardTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render progress dashboard as Markdown.

        Args:
            data: Progress tracking engine result data.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_kpi_cards(data),
            self._md_savings_trend(data),
            self._md_implementation_status(data),
            self._md_variance_analysis(data),
            self._md_alerts(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._generate_provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render progress dashboard as self-contained HTML.

        Args:
            data: Progress tracking engine result data.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_kpi_cards(data),
            self._html_savings_trend(data),
            self._html_implementation_status(data),
            self._html_variance_analysis(data),
            self._html_alerts(data),
        ])
        prov = self._generate_provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Progress Dashboard</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render progress dashboard as structured JSON.

        Args:
            data: Progress tracking engine result data.

        Returns:
            Dict with structured dashboard sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "progress_dashboard",
            "version": "33.0.0",
            "generated_at": self.generated_at.isoformat(),
            "kpis": self._json_kpis(data),
            "savings_trend": data.get("savings_trend", []),
            "implementation_status": data.get("implementation_status", []),
            "variance_analysis": data.get("variance_analysis", []),
            "alerts": data.get("alerts", []),
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
            f"# Quick Wins Progress Dashboard\n\n"
            f"**Facility:** {facility}  \n"
            f"**Reporting Period:** {data.get('reporting_period', '')}  \n"
            f"**Dashboard Generated:** {ts}  \n"
            f"**Template:** PACK-033 ProgressDashboardTemplate v33.0.0\n\n---"
        )

    def _md_kpi_cards(self, data: Dict[str, Any]) -> str:
        """Render KPI cards section."""
        kpis = data.get("kpis", {})
        return (
            "## 1. Key Performance Indicators\n\n"
            "| KPI | Value | Target | Status |\n|-----|-------|--------|--------|\n"
            f"| Total Savings Realized | {self._format_currency(kpis.get('total_savings_realized', 0))} "
            f"| {self._format_currency(kpis.get('total_savings_target', 0))} "
            f"| {kpis.get('savings_status', '-')} |\n"
            f"| ROI | {self._fmt(kpis.get('roi_pct', 0))}% "
            f"| {self._fmt(kpis.get('roi_target_pct', 0))}% "
            f"| {kpis.get('roi_status', '-')} |\n"
            f"| Completion Rate | {self._fmt(kpis.get('completion_rate_pct', 0))}% "
            f"| {self._fmt(kpis.get('completion_target_pct', 0))}% "
            f"| {kpis.get('completion_status', '-')} |\n"
            f"| CO2e Avoided | {self._fmt(kpis.get('co2e_avoided_tonnes', 0))} t "
            f"| {self._fmt(kpis.get('co2e_target_tonnes', 0))} t "
            f"| {kpis.get('co2e_status', '-')} |\n"
            f"| Energy Saved | {self._format_energy(kpis.get('energy_saved_mwh', 0))} "
            f"| {self._format_energy(kpis.get('energy_target_mwh', 0))} "
            f"| {kpis.get('energy_status', '-')} |"
        )

    def _md_savings_trend(self, data: Dict[str, Any]) -> str:
        """Render monthly savings trend section."""
        trend = data.get("savings_trend", [])
        if not trend:
            return "## 2. Savings Trend (Monthly)\n\n_No trend data available._"
        lines = [
            "## 2. Savings Trend (Monthly)\n",
            "| Month | Planned Savings | Actual Savings | Variance | Cumulative |",
            "|-------|----------------|---------------|----------|------------|",
        ]
        for t in trend:
            variance = t.get("actual_savings", 0) - t.get("planned_savings", 0)
            lines.append(
                f"| {t.get('month', '-')} "
                f"| {self._format_currency(t.get('planned_savings', 0))} "
                f"| {self._format_currency(t.get('actual_savings', 0))} "
                f"| {self._format_currency(variance)} "
                f"| {self._format_currency(t.get('cumulative_savings', 0))} |"
            )
        return "\n".join(lines)

    def _md_implementation_status(self, data: Dict[str, Any]) -> str:
        """Render implementation status section."""
        statuses = data.get("implementation_status", [])
        if not statuses:
            return "## 3. Implementation Status\n\n_No status data available._"
        lines = [
            "## 3. Implementation Status\n",
            "| Measure | Status | Progress (%) | On Schedule | Savings Realized |",
            "|---------|--------|-------------|-------------|-----------------|",
        ]
        for s in statuses:
            lines.append(
                f"| {s.get('measure', '-')} "
                f"| {s.get('status', '-')} "
                f"| {self._fmt(s.get('progress_pct', 0), 0)}% "
                f"| {'Yes' if s.get('on_schedule', True) else 'No'} "
                f"| {self._format_currency(s.get('savings_realized', 0))} |"
            )
        return "\n".join(lines)

    def _md_variance_analysis(self, data: Dict[str, Any]) -> str:
        """Render variance analysis section."""
        variances = data.get("variance_analysis", [])
        if not variances:
            return "## 4. Variance Analysis\n\n_No variances to report._"
        lines = [
            "## 4. Variance Analysis\n",
            "| Measure | Expected Savings | Actual Savings | Variance | Root Cause |",
            "|---------|-----------------|---------------|----------|------------|",
        ]
        for v in variances:
            var_amount = v.get("actual_savings", 0) - v.get("expected_savings", 0)
            lines.append(
                f"| {v.get('measure', '-')} "
                f"| {self._format_currency(v.get('expected_savings', 0))} "
                f"| {self._format_currency(v.get('actual_savings', 0))} "
                f"| {self._format_currency(var_amount)} "
                f"| {v.get('root_cause', '-')} |"
            )
        return "\n".join(lines)

    def _md_alerts(self, data: Dict[str, Any]) -> str:
        """Render alerts section."""
        alerts = data.get("alerts", [])
        if not alerts:
            return "## 5. Alerts\n\n_No active alerts._"
        lines = ["## 5. Alerts\n"]
        for a in alerts:
            severity = a.get("severity", "info").upper()
            lines.append(
                f"- **[{severity}]** {a.get('message', '-')} "
                f"(Measure: {a.get('measure', '-')}, "
                f"Date: {a.get('date', '-')})"
            )
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render dashboard footer."""
        return "---\n\n*Generated by GreenLang PACK-033 Quick Wins Identifier Pack*"

    # ------------------------------------------------------------------
    # HTML section renderers
    # ------------------------------------------------------------------

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        facility = data.get("facility_name", "Facility")
        return (
            f'<h1>Quick Wins Progress Dashboard</h1>\n'
            f'<p class="subtitle">Facility: {facility} | '
            f'Period: {data.get("reporting_period", "-")}</p>'
        )

    def _html_kpi_cards(self, data: Dict[str, Any]) -> str:
        """Render HTML KPI cards."""
        kpis = data.get("kpis", {})
        return (
            '<h2>Key Performance Indicators</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Savings Realized</span>'
            f'<span class="value">{self._format_currency(kpis.get("total_savings_realized", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">ROI</span>'
            f'<span class="value">{self._fmt(kpis.get("roi_pct", 0))}%</span></div>\n'
            f'  <div class="card"><span class="label">Completion</span>'
            f'<span class="value">{self._fmt(kpis.get("completion_rate_pct", 0), 0)}%</span></div>\n'
            f'  <div class="card"><span class="label">CO2e Avoided</span>'
            f'<span class="value">{self._fmt(kpis.get("co2e_avoided_tonnes", 0))} t</span></div>\n'
            f'  <div class="card"><span class="label">Energy Saved</span>'
            f'<span class="value">{self._format_energy(kpis.get("energy_saved_mwh", 0))}</span></div>\n'
            '</div>'
        )

    def _html_savings_trend(self, data: Dict[str, Any]) -> str:
        """Render HTML savings trend table."""
        trend = data.get("savings_trend", [])
        rows = ""
        for t in trend:
            rows += (
                f'<tr><td>{t.get("month", "-")}</td>'
                f'<td>{self._format_currency(t.get("planned_savings", 0))}</td>'
                f'<td>{self._format_currency(t.get("actual_savings", 0))}</td>'
                f'<td>{self._format_currency(t.get("cumulative_savings", 0))}</td></tr>\n'
            )
        return (
            '<h2>Savings Trend</h2>\n'
            '<table>\n<tr><th>Month</th><th>Planned</th>'
            f'<th>Actual</th><th>Cumulative</th></tr>\n{rows}</table>'
        )

    def _html_implementation_status(self, data: Dict[str, Any]) -> str:
        """Render HTML implementation status."""
        statuses = data.get("implementation_status", [])
        rows = ""
        for s in statuses:
            progress = s.get("progress_pct", 0)
            cls = "progress-complete" if progress >= 100 else "progress-ongoing"
            rows += (
                f'<tr><td>{s.get("measure", "-")}</td>'
                f'<td>{s.get("status", "-")}</td>'
                f'<td><div class="progress-bar"><div class="{cls}" '
                f'style="width:{min(progress, 100)}%">{self._fmt(progress, 0)}%</div></div></td>'
                f'<td>{"Yes" if s.get("on_schedule", True) else "No"}</td></tr>\n'
            )
        return (
            '<h2>Implementation Status</h2>\n'
            '<table>\n<tr><th>Measure</th><th>Status</th>'
            f'<th>Progress</th><th>On Schedule</th></tr>\n{rows}</table>'
        )

    def _html_variance_analysis(self, data: Dict[str, Any]) -> str:
        """Render HTML variance analysis."""
        variances = data.get("variance_analysis", [])
        rows = ""
        for v in variances:
            var_amount = v.get("actual_savings", 0) - v.get("expected_savings", 0)
            cls = "variance-positive" if var_amount >= 0 else "variance-negative"
            rows += (
                f'<tr><td>{v.get("measure", "-")}</td>'
                f'<td>{self._format_currency(v.get("expected_savings", 0))}</td>'
                f'<td>{self._format_currency(v.get("actual_savings", 0))}</td>'
                f'<td class="{cls}">{self._format_currency(var_amount)}</td></tr>\n'
            )
        return (
            '<h2>Variance Analysis</h2>\n'
            '<table>\n<tr><th>Measure</th><th>Expected</th>'
            f'<th>Actual</th><th>Variance</th></tr>\n{rows}</table>'
        )

    def _html_alerts(self, data: Dict[str, Any]) -> str:
        """Render HTML alerts."""
        alerts = data.get("alerts", [])
        items = ""
        for a in alerts:
            severity = a.get("severity", "info").lower()
            items += (
                f'<div class="alert alert-{severity}">'
                f'<strong>[{severity.upper()}]</strong> {a.get("message", "-")} '
                f'(Measure: {a.get("measure", "-")})</div>\n'
            )
        return f'<h2>Alerts</h2>\n{items}'

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_kpis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON KPIs."""
        kpis = data.get("kpis", {})
        return {
            "total_savings_realized": kpis.get("total_savings_realized", 0),
            "total_savings_target": kpis.get("total_savings_target", 0),
            "roi_pct": kpis.get("roi_pct", 0),
            "completion_rate_pct": kpis.get("completion_rate_pct", 0),
            "co2e_avoided_tonnes": kpis.get("co2e_avoided_tonnes", 0),
            "energy_saved_mwh": kpis.get("energy_saved_mwh", 0),
        }

    def _json_charts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chart data payloads for visualization."""
        trend = data.get("savings_trend", [])
        statuses = data.get("implementation_status", [])
        return {
            "savings_trend_line": {
                "type": "line",
                "labels": [t.get("month", "") for t in trend],
                "series": {
                    "planned": [t.get("planned_savings", 0) for t in trend],
                    "actual": [t.get("actual_savings", 0) for t in trend],
                    "cumulative": [t.get("cumulative_savings", 0) for t in trend],
                },
            },
            "status_donut": {
                "type": "donut",
                "labels": list({s.get("status", "") for s in statuses}),
                "values": [
                    sum(1 for s in statuses if s.get("status") == status)
                    for status in {s.get("status", "") for s in statuses}
                ],
            },
            "progress_bar": {
                "type": "horizontal_bar",
                "labels": [s.get("measure", "") for s in statuses],
                "values": [s.get("progress_pct", 0) for s in statuses],
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
            ".progress-bar{background:#e9ecef;border-radius:4px;overflow:hidden;height:22px;}"
            ".progress-complete{background:#198754;color:#fff;text-align:center;height:100%;line-height:22px;font-size:0.8em;}"
            ".progress-ongoing{background:#0d6efd;color:#fff;text-align:center;height:100%;line-height:22px;font-size:0.8em;}"
            ".variance-positive{color:#198754;font-weight:600;}"
            ".variance-negative{color:#dc3545;font-weight:600;}"
            ".alert{padding:10px 15px;border-radius:4px;margin:5px 0;}"
            ".alert-warning{background:#fff3cd;border-left:4px solid #ffc107;}"
            ".alert-critical{background:#f8d7da;border-left:4px solid #dc3545;}"
            ".alert-info{background:#d1e7dd;border-left:4px solid #198754;}"
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
