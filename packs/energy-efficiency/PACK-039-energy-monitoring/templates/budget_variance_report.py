# -*- coding: utf-8 -*-
"""
BudgetVarianceReportTemplate - Budget vs actual for PACK-039.

Generates comprehensive budget variance reports showing budget vs actual
energy cost comparison with variance decomposition into weather, volume,
and efficiency components, cumulative tracking over fiscal periods, and
forecast-to-completion projections.

Sections:
    1. Variance Overview
    2. Period Detail
    3. Variance Decomposition
    4. Cumulative Tracking
    5. Forecast to Completion
    6. Department Variance
    7. Recommendations

Output Formats:
    - Markdown (with provenance hash comment)
    - HTML (self-contained with inline CSS)
    - JSON (structured with chart data)

Regulatory References:
    - ISO 50001:2018 (Energy management systems - Energy planning)
    - CIMA Management Accounting (Variance analysis methodology)
    - ASHRAE Guideline 14 (Weather normalization)

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


class BudgetVarianceReportTemplate:
    """
    Budget variance report template.

    Renders budget vs actual energy cost reports showing variance
    decomposition into weather, volume, and efficiency components,
    cumulative tracking, forecast-to-completion, and department-level
    breakdowns across markdown, HTML, and JSON formats. All outputs
    include SHA-256 provenance hashing for audit trail integrity.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize BudgetVarianceReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render budget variance report as Markdown.

        Args:
            data: Budget variance engine result data.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_variance_overview(data),
            self._md_period_detail(data),
            self._md_variance_decomposition(data),
            self._md_cumulative_tracking(data),
            self._md_forecast_to_completion(data),
            self._md_department_variance(data),
            self._md_recommendations(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._generate_provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render budget variance report as self-contained HTML.

        Args:
            data: Budget variance engine result data.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_variance_overview(data),
            self._html_period_detail(data),
            self._html_variance_decomposition(data),
            self._html_cumulative_tracking(data),
            self._html_forecast_to_completion(data),
            self._html_department_variance(data),
            self._html_recommendations(data),
        ])
        prov = self._generate_provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Budget Variance Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render budget variance report as structured JSON.

        Args:
            data: Budget variance engine result data.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "budget_variance_report",
            "version": "39.0.0",
            "generated_at": self.generated_at.isoformat(),
            "variance_overview": self._json_variance_overview(data),
            "period_detail": data.get("period_detail", []),
            "variance_decomposition": data.get("variance_decomposition", {}),
            "cumulative_tracking": data.get("cumulative_tracking", []),
            "forecast_to_completion": data.get("forecast_to_completion", {}),
            "department_variance": data.get("department_variance", []),
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
            f"# Budget Variance Report\n\n"
            f"**Facility:** {facility}  \n"
            f"**Fiscal Period:** {data.get('fiscal_period', '')}  \n"
            f"**Budget Total:** {self._format_currency(data.get('budget_total', 0))}  \n"
            f"**Actual Total:** {self._format_currency(data.get('actual_total', 0))}  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-039 BudgetVarianceReportTemplate v39.0.0\n\n---"
        )

    def _md_variance_overview(self, data: Dict[str, Any]) -> str:
        """Render variance overview section."""
        overview = data.get("variance_overview", {})
        budget = data.get("budget_total", 0)
        actual = data.get("actual_total", 0)
        variance = actual - budget
        var_pct = self._pct(abs(variance), budget) if budget > 0 else "0.0%"
        direction = "Over" if variance > 0 else "Under"
        return (
            "## 1. Variance Overview\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| Budget | {self._format_currency(budget)} |\n"
            f"| Actual | {self._format_currency(actual)} |\n"
            f"| Variance | {self._format_currency(variance)} ({direction}) |\n"
            f"| Variance % | {var_pct} |\n"
            f"| Weather Variance | {self._format_currency(overview.get('weather_variance', 0))} |\n"
            f"| Volume Variance | {self._format_currency(overview.get('volume_variance', 0))} |\n"
            f"| Efficiency Variance | {self._format_currency(overview.get('efficiency_variance', 0))} |\n"
            f"| Rate Variance | {self._format_currency(overview.get('rate_variance', 0))} |"
        )

    def _md_period_detail(self, data: Dict[str, Any]) -> str:
        """Render period detail section."""
        periods = data.get("period_detail", [])
        if not periods:
            return "## 2. Period Detail\n\n_No period detail data available._"
        lines = [
            "## 2. Period Detail\n",
            "| Period | Budget | Actual | Variance | Var % | Status |",
            "|--------|-------:|-------:|---------:|------:|--------|",
        ]
        for p in periods:
            budget = p.get("budget", 0)
            actual = p.get("actual", 0)
            variance = actual - budget
            var_pct = ((variance / budget) * 100) if budget != 0 else 0
            status = "Over" if variance > 0 else "Under"
            lines.append(
                f"| {p.get('period', '-')} "
                f"| {self._format_currency(budget)} "
                f"| {self._format_currency(actual)} "
                f"| {self._format_currency(variance)} "
                f"| {self._fmt(var_pct)}% "
                f"| {status} |"
            )
        return "\n".join(lines)

    def _md_variance_decomposition(self, data: Dict[str, Any]) -> str:
        """Render variance decomposition section."""
        decomp = data.get("variance_decomposition", {})
        if not decomp:
            return "## 3. Variance Decomposition\n\n_No decomposition data available._"
        components = decomp.get("components", [])
        lines = [
            "## 3. Variance Decomposition\n",
            "| Component | Variance | % of Total Variance | Direction | Driver |",
            "|-----------|--------:|-------------------:|-----------|--------|",
        ]
        total_var = decomp.get("total_variance", 1)
        for comp in components:
            var = comp.get("variance", 0)
            lines.append(
                f"| {comp.get('component', '-')} "
                f"| {self._format_currency(var)} "
                f"| {self._pct(abs(var), abs(total_var)) if total_var != 0 else '0.0%'} "
                f"| {'Unfavorable' if var > 0 else 'Favorable'} "
                f"| {comp.get('driver', '-')} |"
            )
        return "\n".join(lines)

    def _md_cumulative_tracking(self, data: Dict[str, Any]) -> str:
        """Render cumulative tracking section."""
        tracking = data.get("cumulative_tracking", [])
        if not tracking:
            return "## 4. Cumulative Tracking\n\n_No cumulative tracking data available._"
        lines = [
            "## 4. Cumulative Tracking\n",
            "| Period | Cum. Budget | Cum. Actual | Cum. Variance | Cum. Var % |",
            "|--------|----------:|-----------:|-------------:|----------:|",
        ]
        for t in tracking:
            cb = t.get("cum_budget", 0)
            ca = t.get("cum_actual", 0)
            cv = ca - cb
            cvp = ((cv / cb) * 100) if cb != 0 else 0
            lines.append(
                f"| {t.get('period', '-')} "
                f"| {self._format_currency(cb)} "
                f"| {self._format_currency(ca)} "
                f"| {self._format_currency(cv)} "
                f"| {self._fmt(cvp)}% |"
            )
        return "\n".join(lines)

    def _md_forecast_to_completion(self, data: Dict[str, Any]) -> str:
        """Render forecast to completion section."""
        forecast = data.get("forecast_to_completion", {})
        if not forecast:
            return "## 5. Forecast to Completion\n\n_No forecast data available._"
        return (
            "## 5. Forecast to Completion\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| Annual Budget | {self._format_currency(forecast.get('annual_budget', 0))} |\n"
            f"| YTD Actual | {self._format_currency(forecast.get('ytd_actual', 0))} |\n"
            f"| Remaining Budget | {self._format_currency(forecast.get('remaining_budget', 0))} |\n"
            f"| Forecast Remaining | {self._format_currency(forecast.get('forecast_remaining', 0))} |\n"
            f"| Forecast Total | {self._format_currency(forecast.get('forecast_total', 0))} |\n"
            f"| Projected Variance | {self._format_currency(forecast.get('projected_variance', 0))} |\n"
            f"| Confidence | {self._fmt(forecast.get('confidence_pct', 0))}% |\n"
            f"| Methodology | {forecast.get('methodology', '-')} |"
        )

    def _md_department_variance(self, data: Dict[str, Any]) -> str:
        """Render department variance section."""
        depts = data.get("department_variance", [])
        if not depts:
            return "## 6. Department Variance\n\n_No department variance data available._"
        lines = [
            "## 6. Department Variance\n",
            "| Department | Budget | Actual | Variance | Var % |",
            "|-----------|-------:|-------:|---------:|------:|",
        ]
        for d in depts:
            budget = d.get("budget", 0)
            actual = d.get("actual", 0)
            variance = actual - budget
            var_pct = ((variance / budget) * 100) if budget != 0 else 0
            lines.append(
                f"| {d.get('department', '-')} "
                f"| {self._format_currency(budget)} "
                f"| {self._format_currency(actual)} "
                f"| {self._format_currency(variance)} "
                f"| {self._fmt(var_pct)}% |"
            )
        return "\n".join(lines)

    def _md_recommendations(self, data: Dict[str, Any]) -> str:
        """Render recommendations section."""
        recs = data.get("recommendations", [])
        if not recs:
            recs = [
                "Adjust remaining budget based on weather-normalized forecast",
                "Investigate unfavorable efficiency variances for corrective action",
                "Update budget assumptions for rate changes in upcoming periods",
                "Implement monthly variance review meetings with department heads",
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
            f'<h1>Budget Variance Report</h1>\n'
            f'<p class="subtitle">Facility: {facility} | '
            f'Budget: {self._format_currency(data.get("budget_total", 0))} | '
            f'Period: {data.get("fiscal_period", "-")}</p>'
        )

    def _html_variance_overview(self, data: Dict[str, Any]) -> str:
        """Render HTML variance overview cards."""
        o = data.get("variance_overview", {})
        budget = data.get("budget_total", 0)
        actual = data.get("actual_total", 0)
        variance = actual - budget
        cls = "severity-high" if variance > 0 else "severity-low"
        return (
            '<h2>Variance Overview</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Budget</span>'
            f'<span class="value">{self._format_currency(budget)}</span></div>\n'
            f'  <div class="card"><span class="label">Actual</span>'
            f'<span class="value">{self._format_currency(actual)}</span></div>\n'
            f'  <div class="card"><span class="label">Variance</span>'
            f'<span class="value {cls}">{self._format_currency(variance)}</span></div>\n'
            f'  <div class="card"><span class="label">Weather</span>'
            f'<span class="value">{self._format_currency(o.get("weather_variance", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Efficiency</span>'
            f'<span class="value">{self._format_currency(o.get("efficiency_variance", 0))}</span></div>\n'
            '</div>'
        )

    def _html_period_detail(self, data: Dict[str, Any]) -> str:
        """Render HTML period detail table."""
        periods = data.get("period_detail", [])
        rows = ""
        for p in periods:
            budget = p.get("budget", 0)
            actual = p.get("actual", 0)
            variance = actual - budget
            cls = "severity-high" if variance > 0 else "severity-low"
            rows += (
                f'<tr><td>{p.get("period", "-")}</td>'
                f'<td>{self._format_currency(budget)}</td>'
                f'<td>{self._format_currency(actual)}</td>'
                f'<td class="{cls}">{self._format_currency(variance)}</td></tr>\n'
            )
        return (
            '<h2>Period Detail</h2>\n'
            '<table>\n<tr><th>Period</th><th>Budget</th>'
            f'<th>Actual</th><th>Variance</th></tr>\n{rows}</table>'
        )

    def _html_variance_decomposition(self, data: Dict[str, Any]) -> str:
        """Render HTML variance decomposition table."""
        decomp = data.get("variance_decomposition", {})
        components = decomp.get("components", [])
        rows = ""
        for comp in components:
            var = comp.get("variance", 0)
            cls = "severity-high" if var > 0 else "severity-low"
            rows += (
                f'<tr><td>{comp.get("component", "-")}</td>'
                f'<td class="{cls}">{self._format_currency(var)}</td>'
                f'<td>{"Unfavorable" if var > 0 else "Favorable"}</td>'
                f'<td>{comp.get("driver", "-")}</td></tr>\n'
            )
        return (
            '<h2>Variance Decomposition</h2>\n'
            '<table>\n<tr><th>Component</th><th>Variance</th>'
            f'<th>Direction</th><th>Driver</th></tr>\n{rows}</table>'
        )

    def _html_cumulative_tracking(self, data: Dict[str, Any]) -> str:
        """Render HTML cumulative tracking table."""
        tracking = data.get("cumulative_tracking", [])
        rows = ""
        for t in tracking:
            cb = t.get("cum_budget", 0)
            ca = t.get("cum_actual", 0)
            cv = ca - cb
            rows += (
                f'<tr><td>{t.get("period", "-")}</td>'
                f'<td>{self._format_currency(cb)}</td>'
                f'<td>{self._format_currency(ca)}</td>'
                f'<td>{self._format_currency(cv)}</td></tr>\n'
            )
        return (
            '<h2>Cumulative Tracking</h2>\n'
            '<table>\n<tr><th>Period</th><th>Cum. Budget</th>'
            f'<th>Cum. Actual</th><th>Cum. Variance</th></tr>\n{rows}</table>'
        )

    def _html_forecast_to_completion(self, data: Dict[str, Any]) -> str:
        """Render HTML forecast to completion summary."""
        forecast = data.get("forecast_to_completion", {})
        proj_var = forecast.get("projected_variance", 0)
        cls = "severity-high" if proj_var > 0 else "severity-low"
        return (
            '<h2>Forecast to Completion</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Annual Budget</span>'
            f'<span class="value">{self._format_currency(forecast.get("annual_budget", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Forecast Total</span>'
            f'<span class="value">{self._format_currency(forecast.get("forecast_total", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Projected Variance</span>'
            f'<span class="value {cls}">{self._format_currency(proj_var)}</span></div>\n'
            f'  <div class="card"><span class="label">Confidence</span>'
            f'<span class="value">{self._fmt(forecast.get("confidence_pct", 0))}%</span></div>\n'
            '</div>'
        )

    def _html_department_variance(self, data: Dict[str, Any]) -> str:
        """Render HTML department variance table."""
        depts = data.get("department_variance", [])
        rows = ""
        for d in depts:
            budget = d.get("budget", 0)
            actual = d.get("actual", 0)
            variance = actual - budget
            cls = "severity-high" if variance > 0 else "severity-low"
            rows += (
                f'<tr><td>{d.get("department", "-")}</td>'
                f'<td>{self._format_currency(budget)}</td>'
                f'<td>{self._format_currency(actual)}</td>'
                f'<td class="{cls}">{self._format_currency(variance)}</td></tr>\n'
            )
        return (
            '<h2>Department Variance</h2>\n'
            '<table>\n<tr><th>Department</th><th>Budget</th>'
            f'<th>Actual</th><th>Variance</th></tr>\n{rows}</table>'
        )

    def _html_recommendations(self, data: Dict[str, Any]) -> str:
        """Render HTML recommendations."""
        recs = data.get("recommendations", [
            "Adjust remaining budget based on weather-normalized forecast",
            "Investigate unfavorable efficiency variances for corrective action",
        ])
        items = "".join(f'<li>{r}</li>\n' for r in recs)
        return f'<h2>Recommendations</h2>\n<ol>\n{items}</ol>'

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_variance_overview(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON variance overview."""
        o = data.get("variance_overview", {})
        budget = data.get("budget_total", 0)
        actual = data.get("actual_total", 0)
        return {
            "budget_total": budget,
            "actual_total": actual,
            "total_variance": actual - budget,
            "weather_variance": o.get("weather_variance", 0),
            "volume_variance": o.get("volume_variance", 0),
            "efficiency_variance": o.get("efficiency_variance", 0),
            "rate_variance": o.get("rate_variance", 0),
        }

    def _json_charts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chart data payloads for visualization."""
        periods = data.get("period_detail", [])
        tracking = data.get("cumulative_tracking", [])
        depts = data.get("department_variance", [])
        return {
            "budget_vs_actual": {
                "type": "grouped_bar",
                "labels": [p.get("period", "") for p in periods],
                "series": {
                    "budget": [p.get("budget", 0) for p in periods],
                    "actual": [p.get("actual", 0) for p in periods],
                },
            },
            "cumulative_trend": {
                "type": "dual_line",
                "labels": [t.get("period", "") for t in tracking],
                "series": {
                    "cum_budget": [t.get("cum_budget", 0) for t in tracking],
                    "cum_actual": [t.get("cum_actual", 0) for t in tracking],
                },
            },
            "department_variance": {
                "type": "horizontal_bar",
                "labels": [d.get("department", "") for d in depts],
                "values": [d.get("actual", 0) - d.get("budget", 0) for d in depts],
            },
            "decomposition_waterfall": {
                "type": "waterfall",
                "components": data.get("variance_decomposition", {}).get("components", []),
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
