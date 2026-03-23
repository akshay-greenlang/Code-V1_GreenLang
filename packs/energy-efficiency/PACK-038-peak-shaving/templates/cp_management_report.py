# -*- coding: utf-8 -*-
"""
CPManagementReportTemplate - Coincident peak management for PACK-038.

Generates comprehensive coincident peak (CP) management reports showing
prediction accuracy metrics for system peak forecasts, demand response
performance during CP events, charge allocation impact analysis,
and annual CP day forecasts with confidence intervals.

Sections:
    1. CP Management Summary
    2. Prediction Accuracy
    3. Response Performance
    4. Charge Impact Analysis
    5. Annual CP Forecast
    6. Historical Trends
    7. Recommendations

Output Formats:
    - Markdown (with provenance hash comment)
    - HTML (self-contained with inline CSS)
    - JSON (structured with chart data)

Regulatory References:
    - PJM Reliability Pricing Model (RPM)
    - ERCOT 4CP methodology
    - ISO-NE ICAP tag calculations
    - NYISO UCAP requirements

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


class CPManagementReportTemplate:
    """
    Coincident peak management report template.

    Renders CP management reports showing prediction accuracy,
    response performance during CP events, charge allocation
    impact, and annual forecasts across markdown, HTML, and JSON
    formats. All outputs include SHA-256 provenance hashing for
    audit trail integrity.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize CPManagementReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render CP management report as Markdown.

        Args:
            data: CP management engine result data.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_cp_summary(data),
            self._md_prediction_accuracy(data),
            self._md_response_performance(data),
            self._md_charge_impact(data),
            self._md_annual_forecast(data),
            self._md_historical_trends(data),
            self._md_recommendations(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._generate_provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render CP management report as self-contained HTML.

        Args:
            data: CP management engine result data.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_cp_summary(data),
            self._html_prediction_accuracy(data),
            self._html_response_performance(data),
            self._html_charge_impact(data),
            self._html_annual_forecast(data),
            self._html_historical_trends(data),
            self._html_recommendations(data),
        ])
        prov = self._generate_provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>CP Management Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render CP management report as structured JSON.

        Args:
            data: CP management engine result data.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "cp_management_report",
            "version": "38.0.0",
            "generated_at": self.generated_at.isoformat(),
            "cp_summary": self._json_cp_summary(data),
            "prediction_accuracy": data.get("prediction_accuracy", []),
            "response_performance": data.get("response_performance", []),
            "charge_impact": self._json_charge_impact(data),
            "annual_forecast": data.get("annual_forecast", []),
            "historical_trends": data.get("historical_trends", []),
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
            f"# Coincident Peak Management Report\n\n"
            f"**Facility:** {facility}  \n"
            f"**ISO/RTO:** {data.get('iso_rto', '')}  \n"
            f"**CP Season:** {data.get('cp_season', '')}  \n"
            f"**Peak Demand:** {self._format_power(data.get('peak_demand_kw', 0))}  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-038 CPManagementReportTemplate v38.0.0\n\n---"
        )

    def _md_cp_summary(self, data: Dict[str, Any]) -> str:
        """Render CP management summary section."""
        summary = data.get("cp_summary", {})
        return (
            "## 1. CP Management Summary\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| CP Days Called | {summary.get('cp_days_called', 0)} |\n"
            f"| Successful Responses | {summary.get('successful_responses', 0)} |\n"
            f"| Response Rate | {self._fmt(summary.get('response_rate_pct', 0))}% |\n"
            f"| Avg Load at CP | {self._format_power(summary.get('avg_load_at_cp_kw', 0))} |\n"
            f"| Avg Reduction Achieved | {self._format_power(summary.get('avg_reduction_kw', 0))} |\n"
            f"| Prediction Accuracy | {self._fmt(summary.get('prediction_accuracy_pct', 0))}% |\n"
            f"| Capacity Tag Savings | {self._format_currency(summary.get('capacity_tag_savings', 0))} |\n"
            f"| Annual Charge Avoided | {self._format_currency(summary.get('annual_charge_avoided', 0))} |"
        )

    def _md_prediction_accuracy(self, data: Dict[str, Any]) -> str:
        """Render prediction accuracy section."""
        predictions = data.get("prediction_accuracy", [])
        if not predictions:
            return "## 2. Prediction Accuracy\n\n_No prediction data available._"
        lines = [
            "## 2. Prediction Accuracy\n",
            "| Date | Predicted | Actual | Alert Sent | True CP | Accuracy |",
            "|------|----------|--------|-----------|---------|----------|",
        ]
        for pred in predictions:
            is_cp = "Yes" if pred.get("true_cp", False) else "No"
            alert = "Yes" if pred.get("alert_sent", False) else "No"
            lines.append(
                f"| {pred.get('date', '-')} "
                f"| {self._fmt(pred.get('predicted_probability', 0))}% "
                f"| {self._format_power(pred.get('actual_system_peak_kw', 0))} "
                f"| {alert} "
                f"| {is_cp} "
                f"| {pred.get('accuracy_label', '-')} |"
            )
        return "\n".join(lines)

    def _md_response_performance(self, data: Dict[str, Any]) -> str:
        """Render response performance section."""
        responses = data.get("response_performance", [])
        if not responses:
            return "## 3. Response Performance\n\n_No response data available._"
        lines = [
            "## 3. Response Performance\n",
            "| CP Event | Baseline kW | Actual kW | Reduction kW | % Reduction | Result |",
            "|----------|----------:|--------:|------------:|----------:|--------|",
        ]
        for resp in responses:
            lines.append(
                f"| {resp.get('event_date', '-')} "
                f"| {self._fmt(resp.get('baseline_kw', 0), 1)} "
                f"| {self._fmt(resp.get('actual_kw', 0), 1)} "
                f"| {self._fmt(resp.get('reduction_kw', 0), 1)} "
                f"| {self._fmt(resp.get('reduction_pct', 0))}% "
                f"| {resp.get('result', '-')} |"
            )
        return "\n".join(lines)

    def _md_charge_impact(self, data: Dict[str, Any]) -> str:
        """Render charge impact analysis section."""
        impact = data.get("charge_impact", {})
        if not impact:
            return "## 4. Charge Impact Analysis\n\n_No charge impact data available._"
        return (
            "## 4. Charge Impact Analysis\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| Without Management (Tag) | {self._format_power(impact.get('unmanaged_tag_kw', 0))} |\n"
            f"| With Management (Tag) | {self._format_power(impact.get('managed_tag_kw', 0))} |\n"
            f"| Tag Reduction | {self._format_power(impact.get('tag_reduction_kw', 0))} |\n"
            f"| Capacity Rate | {self._format_currency(impact.get('capacity_rate', 0))}/kW-yr |\n"
            f"| Without Management (Cost) | {self._format_currency(impact.get('unmanaged_cost', 0))} |\n"
            f"| With Management (Cost) | {self._format_currency(impact.get('managed_cost', 0))} |\n"
            f"| Net Savings | {self._format_currency(impact.get('net_savings', 0))} |\n"
            f"| ROI on CP Program | {self._fmt(impact.get('roi_pct', 0))}% |"
        )

    def _md_annual_forecast(self, data: Dict[str, Any]) -> str:
        """Render annual CP forecast section."""
        forecast = data.get("annual_forecast", [])
        if not forecast:
            return "## 5. Annual CP Forecast\n\n_No forecast data available._"
        lines = [
            "## 5. Annual CP Forecast\n",
            "| Month | CP Probability | Expected Peak (kW) | Risk Level | Suggested Action |",
            "|-------|-------------:|------------------:|-----------|-----------------|",
        ]
        for f in forecast:
            lines.append(
                f"| {f.get('month', '-')} "
                f"| {self._fmt(f.get('cp_probability_pct', 0))}% "
                f"| {self._fmt(f.get('expected_peak_kw', 0), 0)} "
                f"| {f.get('risk_level', '-')} "
                f"| {f.get('suggested_action', '-')} |"
            )
        return "\n".join(lines)

    def _md_historical_trends(self, data: Dict[str, Any]) -> str:
        """Render historical CP trends section."""
        trends = data.get("historical_trends", [])
        if not trends:
            return "## 6. Historical Trends\n\n_No historical trend data available._"
        lines = [
            "## 6. Historical Trends\n",
            "| Year | CP Days | Avg Tag (kW) | Capacity Cost | Savings |",
            "|------|-------:|----------:|-------------:|--------:|",
        ]
        for t in trends:
            lines.append(
                f"| {t.get('year', '-')} "
                f"| {t.get('cp_days', 0)} "
                f"| {self._fmt(t.get('avg_tag_kw', 0), 0)} "
                f"| {self._format_currency(t.get('capacity_cost', 0))} "
                f"| {self._format_currency(t.get('savings', 0))} |"
            )
        return "\n".join(lines)

    def _md_recommendations(self, data: Dict[str, Any]) -> str:
        """Render recommendations section."""
        recs = data.get("recommendations", [])
        if not recs:
            recs = [
                "Enhance CP prediction model with weather and load correlations",
                "Pre-position BESS for rapid dispatch on high-probability days",
                "Implement automated curtailment triggers at 80% CP probability",
                "Conduct post-season review to improve prediction accuracy",
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
            f'<h1>Coincident Peak Management Report</h1>\n'
            f'<p class="subtitle">Facility: {facility} | '
            f'ISO/RTO: {data.get("iso_rto", "-")} | '
            f'Season: {data.get("cp_season", "-")}</p>'
        )

    def _html_cp_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML CP summary cards."""
        s = data.get("cp_summary", {})
        return (
            '<h2>CP Management Summary</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">CP Days Called</span>'
            f'<span class="value">{s.get("cp_days_called", 0)}</span></div>\n'
            f'  <div class="card"><span class="label">Response Rate</span>'
            f'<span class="value">{self._fmt(s.get("response_rate_pct", 0))}%</span></div>\n'
            f'  <div class="card"><span class="label">Prediction Accuracy</span>'
            f'<span class="value">{self._fmt(s.get("prediction_accuracy_pct", 0))}%</span></div>\n'
            f'  <div class="card"><span class="label">Tag Savings</span>'
            f'<span class="value">{self._format_currency(s.get("capacity_tag_savings", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Annual Avoided</span>'
            f'<span class="value">{self._format_currency(s.get("annual_charge_avoided", 0))}</span></div>\n'
            '</div>'
        )

    def _html_prediction_accuracy(self, data: Dict[str, Any]) -> str:
        """Render HTML prediction accuracy table."""
        predictions = data.get("prediction_accuracy", [])
        rows = ""
        for pred in predictions:
            is_cp = "Yes" if pred.get("true_cp", False) else "No"
            alert = "Yes" if pred.get("alert_sent", False) else "No"
            rows += (
                f'<tr><td>{pred.get("date", "-")}</td>'
                f'<td>{self._fmt(pred.get("predicted_probability", 0))}%</td>'
                f'<td>{self._format_power(pred.get("actual_system_peak_kw", 0))}</td>'
                f'<td>{alert}</td><td>{is_cp}</td>'
                f'<td>{pred.get("accuracy_label", "-")}</td></tr>\n'
            )
        return (
            '<h2>Prediction Accuracy</h2>\n'
            '<table>\n<tr><th>Date</th><th>Predicted</th><th>Actual Peak</th>'
            f'<th>Alert</th><th>True CP</th><th>Accuracy</th></tr>\n{rows}</table>'
        )

    def _html_response_performance(self, data: Dict[str, Any]) -> str:
        """Render HTML response performance table."""
        responses = data.get("response_performance", [])
        rows = ""
        for resp in responses:
            rows += (
                f'<tr><td>{resp.get("event_date", "-")}</td>'
                f'<td>{self._fmt(resp.get("baseline_kw", 0), 1)}</td>'
                f'<td>{self._fmt(resp.get("actual_kw", 0), 1)}</td>'
                f'<td>{self._fmt(resp.get("reduction_kw", 0), 1)}</td>'
                f'<td>{self._fmt(resp.get("reduction_pct", 0))}%</td>'
                f'<td>{resp.get("result", "-")}</td></tr>\n'
            )
        return (
            '<h2>Response Performance</h2>\n'
            '<table>\n<tr><th>Event</th><th>Baseline kW</th><th>Actual kW</th>'
            f'<th>Reduction</th><th>% Red.</th><th>Result</th></tr>\n{rows}</table>'
        )

    def _html_charge_impact(self, data: Dict[str, Any]) -> str:
        """Render HTML charge impact summary."""
        i = data.get("charge_impact", {})
        return (
            '<h2>Charge Impact</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Unmanaged Tag</span>'
            f'<span class="value">{self._fmt(i.get("unmanaged_tag_kw", 0), 0)} kW</span></div>\n'
            f'  <div class="card"><span class="label">Managed Tag</span>'
            f'<span class="value">{self._fmt(i.get("managed_tag_kw", 0), 0)} kW</span></div>\n'
            f'  <div class="card"><span class="label">Net Savings</span>'
            f'<span class="value">{self._format_currency(i.get("net_savings", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">ROI</span>'
            f'<span class="value">{self._fmt(i.get("roi_pct", 0))}%</span></div>\n'
            '</div>'
        )

    def _html_annual_forecast(self, data: Dict[str, Any]) -> str:
        """Render HTML annual forecast table."""
        forecast = data.get("annual_forecast", [])
        rows = ""
        for f in forecast:
            risk = f.get("risk_level", "low").lower()
            rows += (
                f'<tr><td>{f.get("month", "-")}</td>'
                f'<td>{self._fmt(f.get("cp_probability_pct", 0))}%</td>'
                f'<td>{self._fmt(f.get("expected_peak_kw", 0), 0)}</td>'
                f'<td class="severity-{risk}">{f.get("risk_level", "-")}</td>'
                f'<td>{f.get("suggested_action", "-")}</td></tr>\n'
            )
        return (
            '<h2>Annual CP Forecast</h2>\n'
            '<table>\n<tr><th>Month</th><th>CP Prob.</th><th>Expected Peak</th>'
            f'<th>Risk</th><th>Action</th></tr>\n{rows}</table>'
        )

    def _html_historical_trends(self, data: Dict[str, Any]) -> str:
        """Render HTML historical trends table."""
        trends = data.get("historical_trends", [])
        rows = ""
        for t in trends:
            rows += (
                f'<tr><td>{t.get("year", "-")}</td>'
                f'<td>{t.get("cp_days", 0)}</td>'
                f'<td>{self._fmt(t.get("avg_tag_kw", 0), 0)}</td>'
                f'<td>{self._format_currency(t.get("capacity_cost", 0))}</td>'
                f'<td>{self._format_currency(t.get("savings", 0))}</td></tr>\n'
            )
        return (
            '<h2>Historical Trends</h2>\n'
            '<table>\n<tr><th>Year</th><th>CP Days</th><th>Avg Tag</th>'
            f'<th>Capacity Cost</th><th>Savings</th></tr>\n{rows}</table>'
        )

    def _html_recommendations(self, data: Dict[str, Any]) -> str:
        """Render HTML recommendations."""
        recs = data.get("recommendations", [
            "Enhance CP prediction model with weather correlations",
            "Pre-position BESS for rapid dispatch on high-probability days",
        ])
        items = "".join(f'<li>{r}</li>\n' for r in recs)
        return f'<h2>Recommendations</h2>\n<ol>\n{items}</ol>'

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_cp_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON CP summary."""
        s = data.get("cp_summary", {})
        return {
            "cp_days_called": s.get("cp_days_called", 0),
            "successful_responses": s.get("successful_responses", 0),
            "response_rate_pct": s.get("response_rate_pct", 0),
            "avg_load_at_cp_kw": s.get("avg_load_at_cp_kw", 0),
            "avg_reduction_kw": s.get("avg_reduction_kw", 0),
            "prediction_accuracy_pct": s.get("prediction_accuracy_pct", 0),
            "capacity_tag_savings": s.get("capacity_tag_savings", 0),
            "annual_charge_avoided": s.get("annual_charge_avoided", 0),
        }

    def _json_charge_impact(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON charge impact."""
        i = data.get("charge_impact", {})
        return {
            "unmanaged_tag_kw": i.get("unmanaged_tag_kw", 0),
            "managed_tag_kw": i.get("managed_tag_kw", 0),
            "tag_reduction_kw": i.get("tag_reduction_kw", 0),
            "capacity_rate": i.get("capacity_rate", 0),
            "unmanaged_cost": i.get("unmanaged_cost", 0),
            "managed_cost": i.get("managed_cost", 0),
            "net_savings": i.get("net_savings", 0),
            "roi_pct": i.get("roi_pct", 0),
        }

    def _json_charts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chart data payloads for visualization."""
        predictions = data.get("prediction_accuracy", [])
        responses = data.get("response_performance", [])
        forecast = data.get("annual_forecast", [])
        trends = data.get("historical_trends", [])
        return {
            "prediction_scatter": {
                "type": "scatter",
                "items": [
                    {
                        "date": p.get("date", ""),
                        "probability": p.get("predicted_probability", 0),
                        "true_cp": p.get("true_cp", False),
                    }
                    for p in predictions
                ],
            },
            "response_bar": {
                "type": "grouped_bar",
                "labels": [r.get("event_date", "") for r in responses],
                "series": {
                    "baseline": [r.get("baseline_kw", 0) for r in responses],
                    "actual": [r.get("actual_kw", 0) for r in responses],
                },
            },
            "forecast_heatmap": {
                "type": "heatmap",
                "labels": [f.get("month", "") for f in forecast],
                "values": [f.get("cp_probability_pct", 0) for f in forecast],
            },
            "historical_line": {
                "type": "line",
                "labels": [str(t.get("year", "")) for t in trends],
                "values": [t.get("savings", 0) for t in trends],
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
