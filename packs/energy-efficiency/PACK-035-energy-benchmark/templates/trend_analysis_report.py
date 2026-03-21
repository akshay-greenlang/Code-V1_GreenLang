# -*- coding: utf-8 -*-
"""
TrendAnalysisReportTemplate - Statistical trend analysis report for PACK-035.

Generates statistical trend analysis reports with rolling 12-month EUI
trends, CUSUM analysis for change detection, SPC control chart data,
year-over-year comparisons, Mann-Kendall trend tests, forecasting,
active alerts, and step-change detection.

Sections:
    1. Header
    2. Rolling 12-Month EUI Trend
    3. CUSUM Analysis
    4. SPC Control Chart Data
    5. Year-over-Year Comparison
    6. Mann-Kendall Trend Test
    7. Forecast
    8. Active Alerts
    9. Step Changes Detected
   10. Provenance

Author: GreenLang Team
Version: 35.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class TrendAnalysisReportTemplate:
    """
    Statistical trend analysis report template.

    Renders trend analysis reports with CUSUM, SPC, Mann-Kendall,
    forecasting, and step-change detection across markdown, HTML,
    and JSON formats.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize TrendAnalysisReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render trend analysis report as Markdown.

        Args:
            data: Trend analysis data from engine processing.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_rolling_trend(data),
            self._md_cusum(data),
            self._md_spc(data),
            self._md_yoy_comparison(data),
            self._md_mann_kendall(data),
            self._md_forecast(data),
            self._md_active_alerts(data),
            self._md_step_changes(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render trend analysis report as self-contained HTML.

        Args:
            data: Trend analysis data from engine processing.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_rolling_trend(data),
            self._html_cusum(data),
            self._html_spc(data),
            self._html_yoy_comparison(data),
            self._html_mann_kendall(data),
            self._html_forecast(data),
            self._html_active_alerts(data),
            self._html_step_changes(data),
        ])
        prov = self._provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Trend Analysis Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render trend analysis report as structured JSON.

        Args:
            data: Trend analysis data from engine processing.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "trend_analysis_report",
            "version": "35.0.0",
            "generated_at": self.generated_at.isoformat(),
            "facility": data.get("facility", {}),
            "rolling_trend": data.get("rolling_trend", []),
            "cusum": data.get("cusum", {}),
            "spc": data.get("spc", {}),
            "yoy_comparison": data.get("yoy_comparison", []),
            "mann_kendall": data.get("mann_kendall", {}),
            "forecast": data.get("forecast", {}),
            "active_alerts": data.get("active_alerts", []),
            "step_changes": data.get("step_changes", []),
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
        facility = data.get("facility_name", "Facility")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            "# Energy Trend Analysis Report\n\n"
            f"**Facility:** {facility}  \n"
            f"**Analysis Period:** {data.get('analysis_period', '-')}  \n"
            f"**Data Points:** {data.get('data_point_count', 0)}  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-035 TrendAnalysisReportTemplate v35.0.0\n\n---"
        )

    def _md_rolling_trend(self, data: Dict[str, Any]) -> str:
        """Render rolling 12-month EUI trend section."""
        trend = data.get("rolling_trend", [])
        if not trend:
            return "## 1. Rolling 12-Month EUI Trend\n\n_No trend data._"
        lines = [
            "## 1. Rolling 12-Month EUI Trend\n",
            "| Period End | EUI (kWh/m2) | Change vs Prior | 12-Month Avg | Status |",
            "|-----------|-------------|----------------|-------------|--------|",
        ]
        for t in trend:
            lines.append(
                f"| {t.get('period_end', '-')} "
                f"| {self._fmt(t.get('eui', 0))} "
                f"| {self._fmt(t.get('change_vs_prior', 0))} "
                f"| {self._fmt(t.get('rolling_avg', 0))} "
                f"| {t.get('status', '-')} |"
            )
        return "\n".join(lines)

    def _md_cusum(self, data: Dict[str, Any]) -> str:
        """Render CUSUM analysis section."""
        cusum = data.get("cusum", {})
        points = cusum.get("data_points", [])
        lines = [
            "## 2. CUSUM Analysis\n",
            f"**Target EUI:** {self._fmt(cusum.get('target_eui', 0))} kWh/m2/yr  ",
            f"**Current CUSUM:** {self._fmt(cusum.get('current_cusum', 0))}  ",
            f"**Trend:** {cusum.get('trend', '-')}  ",
            f"**Decision Interval (H):** {self._fmt(cusum.get('decision_interval', 0))}  ",
            f"**Alert Triggered:** {'Yes' if cusum.get('alert', False) else 'No'}",
        ]
        if points:
            lines.extend([
                "\n### CUSUM Data Points\n",
                "| Period | Actual EUI | Deviation | Cumulative Sum |",
                "|--------|-----------|-----------|---------------|",
            ])
            for p in points[-12:]:
                lines.append(
                    f"| {p.get('period', '-')} "
                    f"| {self._fmt(p.get('actual_eui', 0))} "
                    f"| {self._fmt(p.get('deviation', 0))} "
                    f"| {self._fmt(p.get('cusum', 0))} |"
                )
        return "\n".join(lines)

    def _md_spc(self, data: Dict[str, Any]) -> str:
        """Render SPC control chart data section."""
        spc = data.get("spc", {})
        points = spc.get("data_points", [])
        lines = [
            "## 3. SPC Control Chart\n",
            f"**Mean (CL):** {self._fmt(spc.get('mean', 0))} kWh/m2  ",
            f"**Upper Control Limit (UCL):** {self._fmt(spc.get('ucl', 0))}  ",
            f"**Lower Control Limit (LCL):** {self._fmt(spc.get('lcl', 0))}  ",
            f"**Sigma:** {self._fmt(spc.get('sigma', 0))}  ",
            f"**Out-of-Control Points:** {spc.get('out_of_control_count', 0)}  ",
            f"**Western Electric Rules Violated:** {spc.get('we_rules_violated', 0)}",
        ]
        if points:
            lines.extend([
                "\n### Recent Control Points\n",
                "| Period | Value | Zone | Out of Control |",
                "|--------|-------|------|---------------|",
            ])
            for p in points[-12:]:
                ooc = "YES" if p.get("out_of_control", False) else "-"
                lines.append(
                    f"| {p.get('period', '-')} "
                    f"| {self._fmt(p.get('value', 0))} "
                    f"| {p.get('zone', '-')} "
                    f"| {ooc} |"
                )
        return "\n".join(lines)

    def _md_yoy_comparison(self, data: Dict[str, Any]) -> str:
        """Render year-over-year comparison section."""
        yoy = data.get("yoy_comparison", [])
        if not yoy:
            return "## 4. Year-over-Year Comparison\n\n_No YoY data._"
        lines = [
            "## 4. Year-over-Year Comparison\n",
            "| Year | EUI | Change | Change (%) | Energy Cost |",
            "|------|-----|--------|-----------|------------|",
        ]
        for y in yoy:
            lines.append(
                f"| {y.get('year', '-')} "
                f"| {self._fmt(y.get('eui', 0))} "
                f"| {self._fmt(y.get('change', 0))} "
                f"| {self._fmt(y.get('change_pct', 0))}% "
                f"| EUR {self._fmt(y.get('energy_cost', 0))} |"
            )
        return "\n".join(lines)

    def _md_mann_kendall(self, data: Dict[str, Any]) -> str:
        """Render Mann-Kendall trend test section."""
        mk = data.get("mann_kendall", {})
        return (
            "## 5. Mann-Kendall Trend Test\n\n"
            "| Parameter | Value |\n|-----------|-------|\n"
            f"| Trend Direction | {mk.get('trend_direction', '-')} |\n"
            f"| S Statistic | {self._fmt(mk.get('s_statistic', 0), 0)} |\n"
            f"| Z Score | {self._fmt(mk.get('z_score', 0), 3)} |\n"
            f"| P Value | {self._fmt(mk.get('p_value', 0), 4)} |\n"
            f"| Significant (alpha=0.05) | "
            f"{'Yes' if mk.get('significant', False) else 'No'} |\n"
            f"| Sen Slope | {self._fmt(mk.get('sen_slope', 0), 4)} kWh/m2/period |\n"
            f"| Interpretation | {mk.get('interpretation', '-')} |"
        )

    def _md_forecast(self, data: Dict[str, Any]) -> str:
        """Render forecast section."""
        fc = data.get("forecast", {})
        points = fc.get("forecast_points", [])
        lines = [
            "## 6. Forecast\n",
            f"**Method:** {fc.get('method', '-')}  ",
            f"**Horizon:** {fc.get('horizon', '-')}  ",
            f"**Confidence Interval:** {fc.get('confidence_pct', 95)}%",
        ]
        if points:
            lines.extend([
                "\n### Forecast Points\n",
                "| Period | Forecast EUI | Lower Bound | Upper Bound |",
                "|--------|-------------|-------------|------------|",
            ])
            for p in points:
                lines.append(
                    f"| {p.get('period', '-')} "
                    f"| {self._fmt(p.get('forecast', 0))} "
                    f"| {self._fmt(p.get('lower', 0))} "
                    f"| {self._fmt(p.get('upper', 0))} |"
                )
        return "\n".join(lines)

    def _md_active_alerts(self, data: Dict[str, Any]) -> str:
        """Render active alerts section."""
        alerts = data.get("active_alerts", [])
        if not alerts:
            return "## 7. Active Alerts\n\n_No active alerts._"
        lines = [
            "## 7. Active Alerts\n",
            "| # | Alert | Severity | Triggered | Description |",
            "|---|-------|---------|-----------|------------|",
        ]
        for i, a in enumerate(alerts, 1):
            lines.append(
                f"| {i} | {a.get('alert_type', '-')} "
                f"| {a.get('severity', '-')} "
                f"| {a.get('triggered_at', '-')} "
                f"| {a.get('description', '-')} |"
            )
        return "\n".join(lines)

    def _md_step_changes(self, data: Dict[str, Any]) -> str:
        """Render step changes detected section."""
        steps = data.get("step_changes", [])
        if not steps:
            return "## 8. Step Changes Detected\n\n_No step changes detected._"
        lines = [
            "## 8. Step Changes Detected\n",
            "| # | Date | EUI Before | EUI After | Change | Probable Cause |",
            "|---|------|-----------|----------|--------|---------------|",
        ]
        for i, s in enumerate(steps, 1):
            lines.append(
                f"| {i} | {s.get('date', '-')} "
                f"| {self._fmt(s.get('eui_before', 0))} "
                f"| {self._fmt(s.get('eui_after', 0))} "
                f"| {self._fmt(s.get('change', 0))} "
                f"| {s.get('probable_cause', '-')} |"
            )
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render report footer."""
        return "---\n\n*Generated by GreenLang PACK-035 Energy Benchmark Pack*"

    # ------------------------------------------------------------------
    # HTML section renderers
    # ------------------------------------------------------------------

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        facility = data.get("facility_name", "Facility")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<h1>Energy Trend Analysis Report</h1>\n'
            f'<p class="subtitle">Facility: {facility} | '
            f'Period: {data.get("analysis_period", "-")} | '
            f'Generated: {ts}</p>'
        )

    def _html_rolling_trend(self, data: Dict[str, Any]) -> str:
        """Render HTML rolling trend table."""
        trend = data.get("rolling_trend", [])
        rows = "".join(
            f'<tr><td>{t.get("period_end", "-")}</td>'
            f'<td>{self._fmt(t.get("eui", 0))}</td>'
            f'<td>{self._fmt(t.get("change_vs_prior", 0))}</td>'
            f'<td>{self._fmt(t.get("rolling_avg", 0))}</td></tr>\n'
            for t in trend
        )
        return (
            '<h2>Rolling 12-Month EUI Trend</h2>\n'
            '<table>\n<tr><th>Period</th><th>EUI</th>'
            f'<th>Change</th><th>12M Avg</th></tr>\n{rows}</table>'
        )

    def _html_cusum(self, data: Dict[str, Any]) -> str:
        """Render HTML CUSUM analysis."""
        cusum = data.get("cusum", {})
        alert = cusum.get("alert", False)
        alert_cls = "alert-active" if alert else "alert-none"
        return (
            '<h2>CUSUM Analysis</h2>\n'
            f'<div class="info-box">'
            f'<p>Target: {self._fmt(cusum.get("target_eui", 0))} kWh/m2/yr | '
            f'Current CUSUM: {self._fmt(cusum.get("current_cusum", 0))} | '
            f'Trend: {cusum.get("trend", "-")} | '
            f'<span class="{alert_cls}">Alert: {"ACTIVE" if alert else "None"}</span></p>'
            '</div>'
        )

    def _html_spc(self, data: Dict[str, Any]) -> str:
        """Render HTML SPC control chart section."""
        spc = data.get("spc", {})
        return (
            '<h2>SPC Control Chart</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">UCL</span>'
            f'<span class="value">{self._fmt(spc.get("ucl", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Mean (CL)</span>'
            f'<span class="value">{self._fmt(spc.get("mean", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">LCL</span>'
            f'<span class="value">{self._fmt(spc.get("lcl", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Out of Control</span>'
            f'<span class="value">{spc.get("out_of_control_count", 0)}</span></div>\n'
            '</div>'
        )

    def _html_yoy_comparison(self, data: Dict[str, Any]) -> str:
        """Render HTML year-over-year comparison."""
        yoy = data.get("yoy_comparison", [])
        rows = "".join(
            f'<tr><td>{y.get("year", "-")}</td>'
            f'<td>{self._fmt(y.get("eui", 0))}</td>'
            f'<td>{self._fmt(y.get("change_pct", 0))}%</td></tr>\n'
            for y in yoy
        )
        return (
            '<h2>Year-over-Year Comparison</h2>\n'
            '<table>\n<tr><th>Year</th><th>EUI</th>'
            f'<th>Change</th></tr>\n{rows}</table>'
        )

    def _html_mann_kendall(self, data: Dict[str, Any]) -> str:
        """Render HTML Mann-Kendall test results."""
        mk = data.get("mann_kendall", {})
        sig = mk.get("significant", False)
        cls = "status-pass" if mk.get("trend_direction", "") == "decreasing" else "status-fail"
        return (
            '<h2>Mann-Kendall Trend Test</h2>\n'
            f'<div class="info-box">'
            f'<p><strong>Direction:</strong> '
            f'<span class="{cls}">{mk.get("trend_direction", "-")}</span> | '
            f'<strong>P-value:</strong> {self._fmt(mk.get("p_value", 0), 4)} | '
            f'<strong>Significant:</strong> {"Yes" if sig else "No"} | '
            f'<strong>Sen Slope:</strong> {self._fmt(mk.get("sen_slope", 0), 4)}</p>'
            f'<p>{mk.get("interpretation", "")}</p></div>'
        )

    def _html_forecast(self, data: Dict[str, Any]) -> str:
        """Render HTML forecast section."""
        fc = data.get("forecast", {})
        points = fc.get("forecast_points", [])
        rows = "".join(
            f'<tr><td>{p.get("period", "-")}</td>'
            f'<td>{self._fmt(p.get("forecast", 0))}</td>'
            f'<td>{self._fmt(p.get("lower", 0))}</td>'
            f'<td>{self._fmt(p.get("upper", 0))}</td></tr>\n'
            for p in points
        )
        return (
            '<h2>Forecast</h2>\n'
            f'<p>Method: {fc.get("method", "-")} | '
            f'Horizon: {fc.get("horizon", "-")}</p>\n'
            '<table>\n<tr><th>Period</th><th>Forecast</th>'
            f'<th>Lower</th><th>Upper</th></tr>\n{rows}</table>'
        )

    def _html_active_alerts(self, data: Dict[str, Any]) -> str:
        """Render HTML active alerts."""
        alerts = data.get("active_alerts", [])
        items = ""
        for a in alerts:
            sev = a.get("severity", "low").lower()
            cls = "alert-high" if sev == "high" else ("alert-medium" if sev == "medium" else "alert-low")
            items += (
                f'<div class="alert-item {cls}">'
                f'<strong>{a.get("alert_type", "-")}</strong> | '
                f'Severity: {a.get("severity", "-")} | '
                f'{a.get("description", "-")}</div>\n'
            )
        if not items:
            items = '<p>No active alerts.</p>'
        return f'<h2>Active Alerts</h2>\n{items}'

    def _html_step_changes(self, data: Dict[str, Any]) -> str:
        """Render HTML step changes."""
        steps = data.get("step_changes", [])
        rows = "".join(
            f'<tr><td>{s.get("date", "-")}</td>'
            f'<td>{self._fmt(s.get("eui_before", 0))}</td>'
            f'<td>{self._fmt(s.get("eui_after", 0))}</td>'
            f'<td>{self._fmt(s.get("change", 0))}</td>'
            f'<td>{s.get("probable_cause", "-")}</td></tr>\n'
            for s in steps
        )
        return (
            '<h2>Step Changes Detected</h2>\n'
            '<table>\n<tr><th>Date</th><th>Before</th><th>After</th>'
            f'<th>Change</th><th>Probable Cause</th></tr>\n{rows}</table>'
        )

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_charts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chart data payloads for visualization."""
        trend = data.get("rolling_trend", [])
        cusum_pts = data.get("cusum", {}).get("data_points", [])
        spc = data.get("spc", {})
        forecast_pts = data.get("forecast", {}).get("forecast_points", [])
        return {
            "rolling_trend_line": {
                "type": "line",
                "labels": [t.get("period_end", "") for t in trend],
                "series": {
                    "eui": [t.get("eui", 0) for t in trend],
                    "rolling_avg": [t.get("rolling_avg", 0) for t in trend],
                },
            },
            "cusum_line": {
                "type": "line",
                "labels": [p.get("period", "") for p in cusum_pts],
                "values": [p.get("cusum", 0) for p in cusum_pts],
            },
            "spc_control_chart": {
                "type": "control_chart",
                "mean": spc.get("mean", 0),
                "ucl": spc.get("ucl", 0),
                "lcl": spc.get("lcl", 0),
                "labels": [p.get("period", "") for p in spc.get("data_points", [])],
                "values": [p.get("value", 0) for p in spc.get("data_points", [])],
            },
            "forecast_line": {
                "type": "line",
                "labels": [p.get("period", "") for p in forecast_pts],
                "series": {
                    "forecast": [p.get("forecast", 0) for p in forecast_pts],
                    "lower": [p.get("lower", 0) for p in forecast_pts],
                    "upper": [p.get("upper", 0) for p in forecast_pts],
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
            ".info-box{background:#e7f1ff;border-left:4px solid #0d6efd;padding:12px 16px;margin:15px 0;}"
            ".alert-item{padding:10px 15px;margin:6px 0;border-radius:4px;}"
            ".alert-high{background:#f8d7da;border-left:4px solid #dc3545;}"
            ".alert-medium{background:#fff3cd;border-left:4px solid #fd7e14;}"
            ".alert-low{background:#d1e7dd;border-left:4px solid #198754;}"
            ".alert-active{color:#dc3545;font-weight:700;}"
            ".alert-none{color:#198754;}"
            ".status-pass{color:#198754;font-weight:700;}"
            ".status-fail{color:#dc3545;font-weight:700;}"
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

    def _provenance(self, content: str) -> str:
        """Compute SHA-256 provenance hash.

        Args:
            content: Content string to hash.

        Returns:
            SHA-256 hex digest.
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
