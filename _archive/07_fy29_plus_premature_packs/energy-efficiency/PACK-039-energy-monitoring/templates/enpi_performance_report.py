# -*- coding: utf-8 -*-
"""
EnPIPerformanceReportTemplate - EnPI tracking for PACK-039.

Generates comprehensive Energy Performance Indicator (EnPI) reports
showing regression model results, CUSUM charts for cumulative savings,
significance testing, baseline comparison with adjustment factors,
and improvement percentage tracking over reporting periods.

Sections:
    1. EnPI Summary
    2. Baseline Comparison
    3. Regression Model
    4. CUSUM Analysis
    5. Significance Testing
    6. Improvement Tracking
    7. Relevant Variables
    8. Recommendations

Output Formats:
    - Markdown (with provenance hash comment)
    - HTML (self-contained with inline CSS)
    - JSON (structured with chart data)

Regulatory References:
    - ISO 50001:2018 (Energy performance indicators)
    - ISO 50006:2014 (Measuring energy performance using EnPIs and EnBs)
    - ISO 50015:2014 (M&V of energy performance)

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


class EnPIPerformanceReportTemplate:
    """
    EnPI performance tracking report template.

    Renders Energy Performance Indicator reports showing regression model
    results, CUSUM cumulative savings charts, statistical significance
    testing, baseline comparison with adjustment factors, and improvement
    percentage tracking across markdown, HTML, and JSON formats. All
    outputs include SHA-256 provenance hashing for audit trail integrity.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize EnPIPerformanceReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render EnPI performance report as Markdown.

        Args:
            data: EnPI performance engine result data.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_enpi_summary(data),
            self._md_baseline_comparison(data),
            self._md_regression_model(data),
            self._md_cusum_analysis(data),
            self._md_significance_testing(data),
            self._md_improvement_tracking(data),
            self._md_relevant_variables(data),
            self._md_recommendations(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._generate_provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render EnPI performance report as self-contained HTML.

        Args:
            data: EnPI performance engine result data.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_enpi_summary(data),
            self._html_baseline_comparison(data),
            self._html_regression_model(data),
            self._html_cusum_analysis(data),
            self._html_significance_testing(data),
            self._html_improvement_tracking(data),
            self._html_relevant_variables(data),
            self._html_recommendations(data),
        ])
        prov = self._generate_provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>EnPI Performance Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render EnPI performance report as structured JSON.

        Args:
            data: EnPI performance engine result data.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "enpi_performance_report",
            "version": "39.0.0",
            "generated_at": self.generated_at.isoformat(),
            "enpi_summary": self._json_enpi_summary(data),
            "baseline_comparison": data.get("baseline_comparison", {}),
            "regression_model": data.get("regression_model", {}),
            "cusum_analysis": data.get("cusum_analysis", []),
            "significance_testing": data.get("significance_testing", {}),
            "improvement_tracking": data.get("improvement_tracking", []),
            "relevant_variables": data.get("relevant_variables", []),
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
            f"# EnPI Performance Report\n\n"
            f"**Facility:** {facility}  \n"
            f"**Reporting Period:** {data.get('reporting_period', '')}  \n"
            f"**Baseline Period:** {data.get('baseline_period', '')}  \n"
            f"**EnPI Count:** {data.get('enpi_count', 0)}  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-039 EnPIPerformanceReportTemplate v39.0.0\n\n---"
        )

    def _md_enpi_summary(self, data: Dict[str, Any]) -> str:
        """Render EnPI summary section."""
        summary = data.get("enpi_summary", {})
        return (
            "## 1. EnPI Summary\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| Primary EnPI | {summary.get('primary_enpi_name', '-')} |\n"
            f"| Baseline Value | {self._fmt(summary.get('baseline_value', 0), 3)} |\n"
            f"| Current Value | {self._fmt(summary.get('current_value', 0), 3)} |\n"
            f"| Improvement | {self._fmt(summary.get('improvement_pct', 0))}% |\n"
            f"| Energy Savings | {self._format_energy(summary.get('energy_savings_mwh', 0))} |\n"
            f"| Cost Savings | {self._format_currency(summary.get('cost_savings', 0))} |\n"
            f"| Model R-squared | {self._fmt(summary.get('r_squared', 0), 4)} |\n"
            f"| Confidence Level | {self._fmt(summary.get('confidence_level', 0))}% |"
        )

    def _md_baseline_comparison(self, data: Dict[str, Any]) -> str:
        """Render baseline comparison section."""
        baseline = data.get("baseline_comparison", {})
        if not baseline:
            return "## 2. Baseline Comparison\n\n_No baseline comparison data available._"
        adjustments = baseline.get("adjustments", [])
        lines = [
            "## 2. Baseline Comparison\n",
            f"**Baseline Period:** {baseline.get('baseline_period', '-')}  \n"
            f"**Baseline Consumption:** {self._format_energy(baseline.get('baseline_consumption_mwh', 0))}  \n"
            f"**Adjusted Baseline:** {self._format_energy(baseline.get('adjusted_baseline_mwh', 0))}  \n"
            f"**Current Consumption:** {self._format_energy(baseline.get('current_consumption_mwh', 0))}  \n"
            f"**Net Savings:** {self._format_energy(baseline.get('net_savings_mwh', 0))}\n",
        ]
        if adjustments:
            lines.append("### Adjustment Factors\n")
            lines.append("| Factor | Baseline | Current | Adjustment (MWh) |")
            lines.append("|--------|--------:|---------:|-----------------:|")
            for adj in adjustments:
                lines.append(
                    f"| {adj.get('factor', '-')} "
                    f"| {self._fmt(adj.get('baseline_value', 0), 2)} "
                    f"| {self._fmt(adj.get('current_value', 0), 2)} "
                    f"| {self._fmt(adj.get('adjustment_mwh', 0), 2)} |"
                )
        return "\n".join(lines)

    def _md_regression_model(self, data: Dict[str, Any]) -> str:
        """Render regression model section."""
        model = data.get("regression_model", {})
        if not model:
            return "## 3. Regression Model\n\n_No regression model data available._"
        coefficients = model.get("coefficients", [])
        lines = [
            "## 3. Regression Model\n",
            f"**Model Type:** {model.get('model_type', '-')}  \n"
            f"**R-squared:** {self._fmt(model.get('r_squared', 0), 4)}  \n"
            f"**Adjusted R-squared:** {self._fmt(model.get('adj_r_squared', 0), 4)}  \n"
            f"**RMSE:** {self._fmt(model.get('rmse', 0), 2)}  \n"
            f"**CV-RMSE:** {self._fmt(model.get('cv_rmse', 0))}%  \n"
            f"**F-statistic:** {self._fmt(model.get('f_statistic', 0), 2)}  \n"
            f"**p-value:** {self._fmt(model.get('p_value', 0), 6)}\n",
        ]
        if coefficients:
            lines.append("### Coefficients\n")
            lines.append("| Variable | Coefficient | Std Error | t-stat | p-value |")
            lines.append("|----------|----------:|----------:|-------:|--------:|")
            for c in coefficients:
                lines.append(
                    f"| {c.get('variable', '-')} "
                    f"| {self._fmt(c.get('coefficient', 0), 4)} "
                    f"| {self._fmt(c.get('std_error', 0), 4)} "
                    f"| {self._fmt(c.get('t_stat', 0), 3)} "
                    f"| {self._fmt(c.get('p_value', 0), 6)} |"
                )
        return "\n".join(lines)

    def _md_cusum_analysis(self, data: Dict[str, Any]) -> str:
        """Render CUSUM analysis section."""
        cusum = data.get("cusum_analysis", [])
        if not cusum:
            return "## 4. CUSUM Analysis\n\n_No CUSUM data available._"
        lines = [
            "## 4. CUSUM Analysis\n",
            "| Period | Actual (MWh) | Predicted (MWh) | Residual (MWh) | Cumulative Sum |",
            "|--------|----------:|--------------:|-------------:|--------------:|",
        ]
        for point in cusum:
            lines.append(
                f"| {point.get('period', '-')} "
                f"| {self._fmt(point.get('actual_mwh', 0), 1)} "
                f"| {self._fmt(point.get('predicted_mwh', 0), 1)} "
                f"| {self._fmt(point.get('residual_mwh', 0), 1)} "
                f"| {self._fmt(point.get('cumulative_sum', 0), 1)} |"
            )
        return "\n".join(lines)

    def _md_significance_testing(self, data: Dict[str, Any]) -> str:
        """Render significance testing section."""
        testing = data.get("significance_testing", {})
        if not testing:
            return "## 5. Significance Testing\n\n_No significance testing data available._"
        return (
            "## 5. Significance Testing\n\n"
            "| Test | Value |\n|------|-------|\n"
            f"| Test Method | {testing.get('method', '-')} |\n"
            f"| Savings (MWh) | {self._fmt(testing.get('savings_mwh', 0), 1)} |\n"
            f"| Uncertainty (MWh) | {self._fmt(testing.get('uncertainty_mwh', 0), 1)} |\n"
            f"| Relative Precision | {self._fmt(testing.get('relative_precision', 0))}% |\n"
            f"| Confidence Interval | {self._fmt(testing.get('confidence_interval_pct', 0))}% |\n"
            f"| t-statistic | {self._fmt(testing.get('t_statistic', 0), 3)} |\n"
            f"| p-value | {self._fmt(testing.get('p_value', 0), 6)} |\n"
            f"| Statistically Significant | {testing.get('is_significant', '-')} |"
        )

    def _md_improvement_tracking(self, data: Dict[str, Any]) -> str:
        """Render improvement tracking section."""
        tracking = data.get("improvement_tracking", [])
        if not tracking:
            return "## 6. Improvement Tracking\n\n_No improvement tracking data available._"
        lines = [
            "## 6. Improvement Tracking\n",
            "| Period | EnPI Value | Baseline EnPI | Improvement (%) | Cumulative Savings (MWh) |",
            "|--------|----------:|-------------:|---------------:|-----------------------:|",
        ]
        for t in tracking:
            lines.append(
                f"| {t.get('period', '-')} "
                f"| {self._fmt(t.get('enpi_value', 0), 3)} "
                f"| {self._fmt(t.get('baseline_enpi', 0), 3)} "
                f"| {self._fmt(t.get('improvement_pct', 0))}% "
                f"| {self._fmt(t.get('cumulative_savings_mwh', 0), 1)} |"
            )
        return "\n".join(lines)

    def _md_relevant_variables(self, data: Dict[str, Any]) -> str:
        """Render relevant variables section."""
        variables = data.get("relevant_variables", [])
        if not variables:
            return "## 7. Relevant Variables\n\n_No relevant variable data available._"
        lines = [
            "## 7. Relevant Variables\n",
            "| Variable | Unit | Baseline Avg | Current Avg | Change (%) | Correlation |",
            "|----------|------|------------:|-----------:|----------:|----------:|",
        ]
        for v in variables:
            lines.append(
                f"| {v.get('variable', '-')} "
                f"| {v.get('unit', '-')} "
                f"| {self._fmt(v.get('baseline_avg', 0), 2)} "
                f"| {self._fmt(v.get('current_avg', 0), 2)} "
                f"| {self._fmt(v.get('change_pct', 0))}% "
                f"| {self._fmt(v.get('correlation', 0), 3)} |"
            )
        return "\n".join(lines)

    def _md_recommendations(self, data: Dict[str, Any]) -> str:
        """Render recommendations section."""
        recs = data.get("recommendations", [])
        if not recs:
            recs = [
                "Revalidate regression model if CV-RMSE exceeds 25% threshold",
                "Update energy baseline when significant static factor changes occur",
                "Investigate negative CUSUM trends indicating performance degradation",
                "Expand relevant variable monitoring to improve model accuracy",
            ]
        lines = ["## 8. Recommendations\n"]
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
            f'<h1>EnPI Performance Report</h1>\n'
            f'<p class="subtitle">Facility: {facility} | '
            f'Reporting: {data.get("reporting_period", "-")} | '
            f'Baseline: {data.get("baseline_period", "-")}</p>'
        )

    def _html_enpi_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML EnPI summary cards."""
        s = data.get("enpi_summary", {})
        return (
            '<h2>EnPI Summary</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Primary EnPI</span>'
            f'<span class="value">{s.get("primary_enpi_name", "-")}</span></div>\n'
            f'  <div class="card"><span class="label">Improvement</span>'
            f'<span class="value">{self._fmt(s.get("improvement_pct", 0))}%</span></div>\n'
            f'  <div class="card"><span class="label">Energy Savings</span>'
            f'<span class="value">{self._fmt(s.get("energy_savings_mwh", 0), 1)} MWh</span></div>\n'
            f'  <div class="card"><span class="label">Cost Savings</span>'
            f'<span class="value">{self._format_currency(s.get("cost_savings", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">R-squared</span>'
            f'<span class="value">{self._fmt(s.get("r_squared", 0), 4)}</span></div>\n'
            '</div>'
        )

    def _html_baseline_comparison(self, data: Dict[str, Any]) -> str:
        """Render HTML baseline comparison table."""
        baseline = data.get("baseline_comparison", {})
        adjustments = baseline.get("adjustments", [])
        rows = ""
        for adj in adjustments:
            rows += (
                f'<tr><td>{adj.get("factor", "-")}</td>'
                f'<td>{self._fmt(adj.get("baseline_value", 0), 2)}</td>'
                f'<td>{self._fmt(adj.get("current_value", 0), 2)}</td>'
                f'<td>{self._fmt(adj.get("adjustment_mwh", 0), 2)}</td></tr>\n'
            )
        return (
            '<h2>Baseline Comparison</h2>\n'
            '<table>\n<tr><th>Factor</th><th>Baseline</th>'
            f'<th>Current</th><th>Adjustment (MWh)</th></tr>\n{rows}</table>'
        )

    def _html_regression_model(self, data: Dict[str, Any]) -> str:
        """Render HTML regression model table."""
        model = data.get("regression_model", {})
        coefficients = model.get("coefficients", [])
        rows = ""
        for c in coefficients:
            rows += (
                f'<tr><td>{c.get("variable", "-")}</td>'
                f'<td>{self._fmt(c.get("coefficient", 0), 4)}</td>'
                f'<td>{self._fmt(c.get("std_error", 0), 4)}</td>'
                f'<td>{self._fmt(c.get("t_stat", 0), 3)}</td>'
                f'<td>{self._fmt(c.get("p_value", 0), 6)}</td></tr>\n'
            )
        return (
            '<h2>Regression Model</h2>\n'
            f'<p>R-squared: {self._fmt(model.get("r_squared", 0), 4)} | '
            f'CV-RMSE: {self._fmt(model.get("cv_rmse", 0))}%</p>\n'
            '<table>\n<tr><th>Variable</th><th>Coefficient</th><th>Std Error</th>'
            f'<th>t-stat</th><th>p-value</th></tr>\n{rows}</table>'
        )

    def _html_cusum_analysis(self, data: Dict[str, Any]) -> str:
        """Render HTML CUSUM analysis table."""
        cusum = data.get("cusum_analysis", [])
        rows = ""
        for point in cusum:
            rows += (
                f'<tr><td>{point.get("period", "-")}</td>'
                f'<td>{self._fmt(point.get("actual_mwh", 0), 1)}</td>'
                f'<td>{self._fmt(point.get("predicted_mwh", 0), 1)}</td>'
                f'<td>{self._fmt(point.get("residual_mwh", 0), 1)}</td>'
                f'<td>{self._fmt(point.get("cumulative_sum", 0), 1)}</td></tr>\n'
            )
        return (
            '<h2>CUSUM Analysis</h2>\n'
            '<table>\n<tr><th>Period</th><th>Actual (MWh)</th><th>Predicted (MWh)</th>'
            f'<th>Residual (MWh)</th><th>Cumulative Sum</th></tr>\n{rows}</table>'
        )

    def _html_significance_testing(self, data: Dict[str, Any]) -> str:
        """Render HTML significance testing summary."""
        testing = data.get("significance_testing", {})
        sig = testing.get("is_significant", False)
        cls = "severity-low" if sig else "severity-high"
        return (
            '<h2>Significance Testing</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Savings</span>'
            f'<span class="value">{self._fmt(testing.get("savings_mwh", 0), 1)} MWh</span></div>\n'
            f'  <div class="card"><span class="label">Uncertainty</span>'
            f'<span class="value">{self._fmt(testing.get("uncertainty_mwh", 0), 1)} MWh</span></div>\n'
            f'  <div class="card"><span class="label">Rel. Precision</span>'
            f'<span class="value">{self._fmt(testing.get("relative_precision", 0))}%</span></div>\n'
            f'  <div class="card"><span class="label">Significant</span>'
            f'<span class="value {cls}">{sig}</span></div>\n'
            '</div>'
        )

    def _html_improvement_tracking(self, data: Dict[str, Any]) -> str:
        """Render HTML improvement tracking table."""
        tracking = data.get("improvement_tracking", [])
        rows = ""
        for t in tracking:
            rows += (
                f'<tr><td>{t.get("period", "-")}</td>'
                f'<td>{self._fmt(t.get("enpi_value", 0), 3)}</td>'
                f'<td>{self._fmt(t.get("baseline_enpi", 0), 3)}</td>'
                f'<td>{self._fmt(t.get("improvement_pct", 0))}%</td>'
                f'<td>{self._fmt(t.get("cumulative_savings_mwh", 0), 1)}</td></tr>\n'
            )
        return (
            '<h2>Improvement Tracking</h2>\n'
            '<table>\n<tr><th>Period</th><th>EnPI Value</th><th>Baseline EnPI</th>'
            f'<th>Improvement (%)</th><th>Cumulative Savings (MWh)</th></tr>\n{rows}</table>'
        )

    def _html_relevant_variables(self, data: Dict[str, Any]) -> str:
        """Render HTML relevant variables table."""
        variables = data.get("relevant_variables", [])
        rows = ""
        for v in variables:
            rows += (
                f'<tr><td>{v.get("variable", "-")}</td>'
                f'<td>{v.get("unit", "-")}</td>'
                f'<td>{self._fmt(v.get("baseline_avg", 0), 2)}</td>'
                f'<td>{self._fmt(v.get("current_avg", 0), 2)}</td>'
                f'<td>{self._fmt(v.get("change_pct", 0))}%</td>'
                f'<td>{self._fmt(v.get("correlation", 0), 3)}</td></tr>\n'
            )
        return (
            '<h2>Relevant Variables</h2>\n'
            '<table>\n<tr><th>Variable</th><th>Unit</th><th>Baseline Avg</th>'
            f'<th>Current Avg</th><th>Change (%)</th><th>Correlation</th></tr>\n{rows}</table>'
        )

    def _html_recommendations(self, data: Dict[str, Any]) -> str:
        """Render HTML recommendations."""
        recs = data.get("recommendations", [
            "Revalidate regression model if CV-RMSE exceeds 25% threshold",
            "Update energy baseline when significant static factor changes occur",
        ])
        items = "".join(f'<li>{r}</li>\n' for r in recs)
        return f'<h2>Recommendations</h2>\n<ol>\n{items}</ol>'

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_enpi_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON EnPI summary."""
        s = data.get("enpi_summary", {})
        return {
            "primary_enpi_name": s.get("primary_enpi_name", ""),
            "baseline_value": s.get("baseline_value", 0),
            "current_value": s.get("current_value", 0),
            "improvement_pct": s.get("improvement_pct", 0),
            "energy_savings_mwh": s.get("energy_savings_mwh", 0),
            "cost_savings": s.get("cost_savings", 0),
            "r_squared": s.get("r_squared", 0),
            "confidence_level": s.get("confidence_level", 0),
        }

    def _json_charts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chart data payloads for visualization."""
        cusum = data.get("cusum_analysis", [])
        tracking = data.get("improvement_tracking", [])
        variables = data.get("relevant_variables", [])
        return {
            "cusum_chart": {
                "type": "line",
                "labels": [c.get("period", "") for c in cusum],
                "values": [c.get("cumulative_sum", 0) for c in cusum],
            },
            "improvement_trend": {
                "type": "bar",
                "labels": [t.get("period", "") for t in tracking],
                "values": [t.get("improvement_pct", 0) for t in tracking],
            },
            "actual_vs_predicted": {
                "type": "dual_line",
                "labels": [c.get("period", "") for c in cusum],
                "series": {
                    "actual": [c.get("actual_mwh", 0) for c in cusum],
                    "predicted": [c.get("predicted_mwh", 0) for c in cusum],
                },
            },
            "variable_correlation": {
                "type": "bar",
                "labels": [v.get("variable", "") for v in variables],
                "values": [v.get("correlation", 0) for v in variables],
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
