# -*- coding: utf-8 -*-
"""
BaselineReportTemplate - Baseline Analysis Report for PACK-040.

Generates comprehensive baseline analysis reports covering regression
model results, statistical validation metrics (CVRMSE, NMBE, R-squared),
independent variable analysis, residual diagnostics, model selection
rationale, and change-point model details.

Sections:
    1. Baseline Summary
    2. Model Selection
    3. Regression Results
    4. Model Validation
    5. Independent Variables
    6. Residual Diagnostics
    7. Change-Point Analysis
    8. Weather Normalization
    9. Baseline Adjustments
    10. Recommendations

Output Formats:
    - Markdown (with provenance hash comment)
    - HTML (self-contained with inline CSS)
    - JSON (structured with chart data)

Regulatory References:
    - ASHRAE Guideline 14-2014 (statistical criteria)
    - ISO 50006:2014 (Energy baselines)
    - IPMVP Core Concepts 2022 (baseline requirements)
    - FEMP M&V Guidelines 4.0

Author: GreenLang Team
Version: 40.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODULE_VERSION = "1.0.0"


class BaselineReportTemplate:
    """
    Baseline analysis report template.

    Renders comprehensive baseline analysis reports showing regression
    model results with coefficient analysis, ASHRAE 14 statistical
    validation (CVRMSE, NMBE, R-squared), independent variable
    significance, residual diagnostics, change-point model details,
    and model selection rationale across markdown, HTML, and JSON
    formats. All outputs include SHA-256 provenance hashing for audit
    trail integrity.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize BaselineReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render baseline analysis report as Markdown.

        Args:
            data: Baseline engine result data.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_baseline_summary(data),
            self._md_model_selection(data),
            self._md_regression_results(data),
            self._md_model_validation(data),
            self._md_independent_variables(data),
            self._md_residual_diagnostics(data),
            self._md_change_point_analysis(data),
            self._md_weather_normalization(data),
            self._md_baseline_adjustments(data),
            self._md_recommendations(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._generate_provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render baseline analysis report as self-contained HTML.

        Args:
            data: Baseline engine result data.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_baseline_summary(data),
            self._html_model_selection(data),
            self._html_regression_results(data),
            self._html_model_validation(data),
            self._html_independent_variables(data),
            self._html_residual_diagnostics(data),
            self._html_change_point_analysis(data),
            self._html_weather_normalization(data),
            self._html_baseline_adjustments(data),
            self._html_recommendations(data),
        ])
        prov = self._generate_provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Baseline Analysis Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render baseline analysis report as structured JSON.

        Args:
            data: Baseline engine result data.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "baseline_report",
            "version": "40.0.0",
            "generated_at": self.generated_at.isoformat(),
            "baseline_summary": self._json_baseline_summary(data),
            "model_selection": data.get("model_selection", {}),
            "regression_results": data.get("regression_results", {}),
            "model_validation": data.get("model_validation", {}),
            "independent_variables": data.get("independent_variables", []),
            "residual_diagnostics": data.get("residual_diagnostics", {}),
            "change_point_analysis": data.get("change_point_analysis", {}),
            "weather_normalization": data.get("weather_normalization", {}),
            "baseline_adjustments": data.get("baseline_adjustments", []),
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
        """Render markdown header with project metadata."""
        facility = data.get("facility_name", "Facility")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Baseline Analysis Report\n\n"
            f"**Facility:** {facility}  \n"
            f"**Baseline Period:** {data.get('baseline_period', '-')}  \n"
            f"**Model Type:** {data.get('model_type', '-')}  \n"
            f"**ECM:** {data.get('ecm_name', '-')}  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-040 BaselineReportTemplate v40.0.0\n\n---"
        )

    def _md_baseline_summary(self, data: Dict[str, Any]) -> str:
        """Render baseline summary section."""
        summary = data.get("baseline_summary", {})
        return (
            "## 1. Baseline Summary\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| Baseline Period | {summary.get('period', '-')} |\n"
            f"| Duration | {summary.get('duration_months', 0)} months |\n"
            f"| Data Points | {summary.get('data_points', 0)} |\n"
            f"| Data Interval | {summary.get('data_interval', '-')} |\n"
            f"| Total Consumption | {self._format_energy(summary.get('total_consumption_mwh', 0))} |\n"
            f"| Average Consumption | {self._format_energy(summary.get('avg_consumption_mwh', 0))} |\n"
            f"| Std Deviation | {self._fmt(summary.get('std_dev_mwh', 0), 2)} MWh |\n"
            f"| Data Completeness | {self._fmt(summary.get('data_completeness_pct', 0))}% |\n"
            f"| Outliers Removed | {summary.get('outliers_removed', 0)} |"
        )

    def _md_model_selection(self, data: Dict[str, Any]) -> str:
        """Render model selection section."""
        selection = data.get("model_selection", {})
        if not selection:
            return "## 2. Model Selection\n\n_No model selection data available._"
        candidates = selection.get("candidates", [])
        lines = [
            "## 2. Model Selection\n",
            f"**Selected Model:** {selection.get('selected_model', '-')}  \n"
            f"**Selection Rationale:** {selection.get('rationale', '-')}  \n"
            f"**Selection Method:** {selection.get('method', '-')}  \n",
        ]
        if candidates:
            lines.append("### Candidate Models\n")
            lines.append("| Model | R-sq | CVRMSE (%) | NMBE (%) | AIC | BIC | Rank |")
            lines.append("|-------|-----:|---------:|---------:|----:|----:|-----:|")
            for c in candidates:
                lines.append(
                    f"| {c.get('model', '-')} "
                    f"| {self._fmt(c.get('r_squared', 0), 4)} "
                    f"| {self._fmt(c.get('cvrmse_pct', 0), 1)} "
                    f"| {self._fmt(c.get('nmbe_pct', 0), 1)} "
                    f"| {self._fmt(c.get('aic', 0), 1)} "
                    f"| {self._fmt(c.get('bic', 0), 1)} "
                    f"| {c.get('rank', '-')} |"
                )
        return "\n".join(lines)

    def _md_regression_results(self, data: Dict[str, Any]) -> str:
        """Render regression results section."""
        results = data.get("regression_results", {})
        if not results:
            return "## 3. Regression Results\n\n_No regression results available._"
        coefficients = results.get("coefficients", [])
        lines = [
            "## 3. Regression Results\n",
            f"**Model Type:** {results.get('model_type', '-')}  \n"
            f"**R-squared:** {self._fmt(results.get('r_squared', 0), 4)}  \n"
            f"**Adjusted R-squared:** {self._fmt(results.get('adj_r_squared', 0), 4)}  \n"
            f"**RMSE:** {self._fmt(results.get('rmse', 0), 3)}  \n"
            f"**F-statistic:** {self._fmt(results.get('f_statistic', 0), 2)}  \n"
            f"**F p-value:** {self._fmt(results.get('f_p_value', 0), 6)}  \n"
            f"**Degrees of Freedom:** {results.get('degrees_of_freedom', '-')}  \n"
            f"**Observations:** {results.get('observations', 0)}\n",
        ]
        if coefficients:
            lines.append("### Coefficients\n")
            lines.append("| Variable | Coefficient | Std Error | t-stat | p-value | Significant |")
            lines.append("|----------|----------:|----------:|-------:|--------:|:-----------:|")
            for c in coefficients:
                sig = "Yes" if c.get("p_value", 1) < 0.05 else "No"
                lines.append(
                    f"| {c.get('variable', '-')} "
                    f"| {self._fmt(c.get('coefficient', 0), 4)} "
                    f"| {self._fmt(c.get('std_error', 0), 4)} "
                    f"| {self._fmt(c.get('t_stat', 0), 3)} "
                    f"| {self._fmt(c.get('p_value', 0), 6)} "
                    f"| {sig} |"
                )
        return "\n".join(lines)

    def _md_model_validation(self, data: Dict[str, Any]) -> str:
        """Render model validation section."""
        validation = data.get("model_validation", {})
        if not validation:
            return "## 4. Model Validation\n\n_No model validation data available._"
        cvrmse = validation.get("cvrmse_pct", 0)
        nmbe = validation.get("nmbe_pct", 0)
        r_sq = validation.get("r_squared", 0)
        cvrmse_pass = "PASS" if cvrmse <= validation.get("cvrmse_threshold_pct", 25) else "FAIL"
        nmbe_pass = "PASS" if abs(nmbe) <= validation.get("nmbe_threshold_pct", 0.5) else "FAIL"
        r_sq_pass = "PASS" if r_sq >= validation.get("r_squared_threshold", 0.7) else "FAIL"
        return (
            "## 4. Model Validation (ASHRAE Guideline 14)\n\n"
            "| Criterion | Value | Threshold | Result |\n|-----------|------:|----------:|:------:|\n"
            f"| CV(RMSE) | {self._fmt(cvrmse, 1)}% | <= {self._fmt(validation.get('cvrmse_threshold_pct', 25), 0)}% | {cvrmse_pass} |\n"
            f"| NMBE | {self._fmt(nmbe, 1)}% | <= +/- {self._fmt(validation.get('nmbe_threshold_pct', 0.5), 1)}% | {nmbe_pass} |\n"
            f"| R-squared | {self._fmt(r_sq, 4)} | >= {self._fmt(validation.get('r_squared_threshold', 0.7), 2)} | {r_sq_pass} |\n"
            f"| DW Statistic | {self._fmt(validation.get('durbin_watson', 0), 3)} | 1.5-2.5 | {validation.get('dw_result', '-')} |\n"
            f"| Normality (SW) | {self._fmt(validation.get('shapiro_wilk_p', 0), 4)} | > 0.05 | {validation.get('normality_result', '-')} |\n"
            f"| Autocorrelation | {self._fmt(validation.get('autocorrelation_lag1', 0), 3)} | < 0.5 | {validation.get('autocorrelation_result', '-')} |\n"
            f"| Overall | - | - | {validation.get('overall_result', '-')} |"
        )

    def _md_independent_variables(self, data: Dict[str, Any]) -> str:
        """Render independent variables section."""
        variables = data.get("independent_variables", [])
        if not variables:
            return "## 5. Independent Variables\n\n_No independent variable data available._"
        lines = [
            "## 5. Independent Variables\n",
            "| Variable | Unit | Mean | Std Dev | Min | Max | Correlation | Significant |",
            "|----------|------|-----:|--------:|----:|----:|----------:|:-----------:|",
        ]
        for v in variables:
            sig = "Yes" if abs(v.get("correlation", 0)) > 0.3 else "No"
            lines.append(
                f"| {v.get('variable', '-')} "
                f"| {v.get('unit', '-')} "
                f"| {self._fmt(v.get('mean', 0), 2)} "
                f"| {self._fmt(v.get('std_dev', 0), 2)} "
                f"| {self._fmt(v.get('min', 0), 2)} "
                f"| {self._fmt(v.get('max', 0), 2)} "
                f"| {self._fmt(v.get('correlation', 0), 3)} "
                f"| {sig} |"
            )
        return "\n".join(lines)

    def _md_residual_diagnostics(self, data: Dict[str, Any]) -> str:
        """Render residual diagnostics section."""
        diag = data.get("residual_diagnostics", {})
        if not diag:
            return "## 6. Residual Diagnostics\n\n_No residual diagnostics data available._"
        return (
            "## 6. Residual Diagnostics\n\n"
            "| Diagnostic | Value | Status |\n|------------|------:|:------:|\n"
            f"| Mean Residual | {self._fmt(diag.get('mean_residual', 0), 4)} | {diag.get('mean_status', '-')} |\n"
            f"| Std Dev Residual | {self._fmt(diag.get('std_dev_residual', 0), 4)} | - |\n"
            f"| Skewness | {self._fmt(diag.get('skewness', 0), 3)} | {diag.get('skewness_status', '-')} |\n"
            f"| Kurtosis | {self._fmt(diag.get('kurtosis', 0), 3)} | {diag.get('kurtosis_status', '-')} |\n"
            f"| Durbin-Watson | {self._fmt(diag.get('durbin_watson', 0), 3)} | {diag.get('dw_status', '-')} |\n"
            f"| Breusch-Pagan p | {self._fmt(diag.get('breusch_pagan_p', 0), 4)} | {diag.get('heteroscedasticity_status', '-')} |\n"
            f"| Max Residual | {self._fmt(diag.get('max_residual', 0), 2)} | - |\n"
            f"| Min Residual | {self._fmt(diag.get('min_residual', 0), 2)} | - |"
        )

    def _md_change_point_analysis(self, data: Dict[str, Any]) -> str:
        """Render change-point analysis section."""
        cp = data.get("change_point_analysis", {})
        if not cp:
            return "## 7. Change-Point Analysis\n\n_No change-point analysis data available._"
        segments = cp.get("segments", [])
        lines = [
            "## 7. Change-Point Analysis\n",
            f"**Model Type:** {cp.get('model_type', '-')}  \n"
            f"**Change Point(s):** {cp.get('change_points', '-')}  \n"
            f"**Balance Point:** {self._fmt(cp.get('balance_point', 0), 1)} deg  \n",
        ]
        if segments:
            lines.append("### Segments\n")
            lines.append("| Segment | Range | Slope | Intercept | R-sq |")
            lines.append("|---------|-------|------:|----------:|-----:|")
            for seg in segments:
                lines.append(
                    f"| {seg.get('name', '-')} "
                    f"| {seg.get('range', '-')} "
                    f"| {self._fmt(seg.get('slope', 0), 4)} "
                    f"| {self._fmt(seg.get('intercept', 0), 2)} "
                    f"| {self._fmt(seg.get('r_squared', 0), 4)} |"
                )
        return "\n".join(lines)

    def _md_weather_normalization(self, data: Dict[str, Any]) -> str:
        """Render weather normalization section."""
        weather = data.get("weather_normalization", {})
        if not weather:
            return "## 8. Weather Normalization\n\n_No weather normalization data available._"
        return (
            "## 8. Weather Normalization\n\n"
            "| Parameter | Value |\n|-----------|-------|\n"
            f"| Weather Station | {weather.get('weather_station', '-')} |\n"
            f"| TMY Source | {weather.get('tmy_source', '-')} |\n"
            f"| HDD Base (C) | {self._fmt(weather.get('hdd_base_c', 0), 1)} |\n"
            f"| CDD Base (C) | {self._fmt(weather.get('cdd_base_c', 0), 1)} |\n"
            f"| Baseline HDD | {self._fmt(weather.get('baseline_hdd', 0), 0)} |\n"
            f"| Baseline CDD | {self._fmt(weather.get('baseline_cdd', 0), 0)} |\n"
            f"| TMY HDD | {self._fmt(weather.get('tmy_hdd', 0), 0)} |\n"
            f"| TMY CDD | {self._fmt(weather.get('tmy_cdd', 0), 0)} |\n"
            f"| Normalized Consumption | {self._format_energy(weather.get('normalized_mwh', 0))} |"
        )

    def _md_baseline_adjustments(self, data: Dict[str, Any]) -> str:
        """Render baseline adjustments section."""
        adjustments = data.get("baseline_adjustments", [])
        if not adjustments:
            return "## 9. Baseline Adjustments\n\n_No baseline adjustment data available._"
        lines = [
            "## 9. Baseline Adjustments\n",
            "| Adjustment | Type | Impact (MWh) | Justification | Date |",
            "|------------|------|----------:|---------------|------|",
        ]
        for adj in adjustments:
            lines.append(
                f"| {adj.get('name', '-')} "
                f"| {adj.get('type', '-')} "
                f"| {self._fmt(adj.get('impact_mwh', 0), 1)} "
                f"| {adj.get('justification', '-')} "
                f"| {adj.get('date', '-')} |"
            )
        return "\n".join(lines)

    def _md_recommendations(self, data: Dict[str, Any]) -> str:
        """Render recommendations section."""
        recs = data.get("recommendations", [])
        if not recs:
            recs = [
                "Verify CV(RMSE) remains below 25% for monthly data per ASHRAE 14",
                "Monitor residual patterns for signs of model degradation",
                "Update baseline model if significant static factors change",
                "Re-evaluate change-point model if occupancy patterns shift",
            ]
        lines = ["## 10. Recommendations\n"]
        for i, rec in enumerate(recs, 1):
            lines.append(f"{i}. {rec}")
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render report footer."""
        return "---\n\n*Generated by GreenLang PACK-040 M&V Pack*"

    # ------------------------------------------------------------------
    # HTML section renderers
    # ------------------------------------------------------------------

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        facility = data.get("facility_name", "Facility")
        return (
            f'<h1>Baseline Analysis Report</h1>\n'
            f'<p class="subtitle">Facility: {facility} | '
            f'Baseline Period: {data.get("baseline_period", "-")} | '
            f'Model: {data.get("model_type", "-")}</p>'
        )

    def _html_baseline_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML baseline summary cards."""
        s = data.get("baseline_summary", {})
        return (
            '<h2>1. Baseline Summary</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Period</span>'
            f'<span class="value">{s.get("period", "-")}</span></div>\n'
            f'  <div class="card"><span class="label">Data Points</span>'
            f'<span class="value">{s.get("data_points", 0)}</span></div>\n'
            f'  <div class="card"><span class="label">Total</span>'
            f'<span class="value">{self._fmt(s.get("total_consumption_mwh", 0), 1)} MWh</span></div>\n'
            f'  <div class="card"><span class="label">Completeness</span>'
            f'<span class="value">{self._fmt(s.get("data_completeness_pct", 0))}%</span></div>\n'
            '</div>'
        )

    def _html_model_selection(self, data: Dict[str, Any]) -> str:
        """Render HTML model selection table."""
        selection = data.get("model_selection", {})
        candidates = selection.get("candidates", [])
        rows = ""
        for c in candidates:
            rows += (
                f'<tr><td>{c.get("model", "-")}</td>'
                f'<td>{self._fmt(c.get("r_squared", 0), 4)}</td>'
                f'<td>{self._fmt(c.get("cvrmse_pct", 0), 1)}%</td>'
                f'<td>{self._fmt(c.get("nmbe_pct", 0), 1)}%</td>'
                f'<td>{c.get("rank", "-")}</td></tr>\n'
            )
        return (
            '<h2>2. Model Selection</h2>\n'
            f'<p>Selected: {selection.get("selected_model", "-")} - '
            f'{selection.get("rationale", "-")}</p>\n'
            '<table>\n<tr><th>Model</th><th>R-sq</th><th>CVRMSE</th>'
            f'<th>NMBE</th><th>Rank</th></tr>\n{rows}</table>'
        )

    def _html_regression_results(self, data: Dict[str, Any]) -> str:
        """Render HTML regression results table."""
        results = data.get("regression_results", {})
        coefficients = results.get("coefficients", [])
        rows = ""
        for c in coefficients:
            sig_cls = "severity-low" if c.get("p_value", 1) < 0.05 else "severity-high"
            rows += (
                f'<tr><td>{c.get("variable", "-")}</td>'
                f'<td>{self._fmt(c.get("coefficient", 0), 4)}</td>'
                f'<td>{self._fmt(c.get("std_error", 0), 4)}</td>'
                f'<td>{self._fmt(c.get("t_stat", 0), 3)}</td>'
                f'<td class="{sig_cls}">{self._fmt(c.get("p_value", 0), 6)}</td></tr>\n'
            )
        return (
            '<h2>3. Regression Results</h2>\n'
            f'<p>R-squared: {self._fmt(results.get("r_squared", 0), 4)} | '
            f'Adj R-sq: {self._fmt(results.get("adj_r_squared", 0), 4)} | '
            f'RMSE: {self._fmt(results.get("rmse", 0), 3)}</p>\n'
            '<table>\n<tr><th>Variable</th><th>Coefficient</th><th>Std Error</th>'
            f'<th>t-stat</th><th>p-value</th></tr>\n{rows}</table>'
        )

    def _html_model_validation(self, data: Dict[str, Any]) -> str:
        """Render HTML model validation cards."""
        v = data.get("model_validation", {})
        cvrmse_cls = "severity-low" if v.get("cvrmse_pct", 100) <= 25 else "severity-high"
        nmbe_cls = "severity-low" if abs(v.get("nmbe_pct", 100)) <= 0.5 else "severity-high"
        rsq_cls = "severity-low" if v.get("r_squared", 0) >= 0.7 else "severity-high"
        return (
            '<h2>4. Model Validation (ASHRAE 14)</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">CV(RMSE)</span>'
            f'<span class="value {cvrmse_cls}">{self._fmt(v.get("cvrmse_pct", 0), 1)}%</span></div>\n'
            f'  <div class="card"><span class="label">NMBE</span>'
            f'<span class="value {nmbe_cls}">{self._fmt(v.get("nmbe_pct", 0), 1)}%</span></div>\n'
            f'  <div class="card"><span class="label">R-squared</span>'
            f'<span class="value {rsq_cls}">{self._fmt(v.get("r_squared", 0), 4)}</span></div>\n'
            f'  <div class="card"><span class="label">Durbin-Watson</span>'
            f'<span class="value">{self._fmt(v.get("durbin_watson", 0), 3)}</span></div>\n'
            f'  <div class="card"><span class="label">Overall</span>'
            f'<span class="value">{v.get("overall_result", "-")}</span></div>\n'
            '</div>'
        )

    def _html_independent_variables(self, data: Dict[str, Any]) -> str:
        """Render HTML independent variables table."""
        variables = data.get("independent_variables", [])
        rows = ""
        for v in variables:
            rows += (
                f'<tr><td>{v.get("variable", "-")}</td>'
                f'<td>{v.get("unit", "-")}</td>'
                f'<td>{self._fmt(v.get("mean", 0), 2)}</td>'
                f'<td>{self._fmt(v.get("std_dev", 0), 2)}</td>'
                f'<td>{self._fmt(v.get("correlation", 0), 3)}</td></tr>\n'
            )
        return (
            '<h2>5. Independent Variables</h2>\n'
            '<table>\n<tr><th>Variable</th><th>Unit</th><th>Mean</th>'
            f'<th>Std Dev</th><th>Correlation</th></tr>\n{rows}</table>'
        )

    def _html_residual_diagnostics(self, data: Dict[str, Any]) -> str:
        """Render HTML residual diagnostics."""
        diag = data.get("residual_diagnostics", {})
        return (
            '<h2>6. Residual Diagnostics</h2>\n'
            '<table>\n'
            '<tr><th>Diagnostic</th><th>Value</th><th>Status</th></tr>\n'
            f'<tr><td>Mean Residual</td><td>{self._fmt(diag.get("mean_residual", 0), 4)}</td>'
            f'<td>{diag.get("mean_status", "-")}</td></tr>\n'
            f'<tr><td>Skewness</td><td>{self._fmt(diag.get("skewness", 0), 3)}</td>'
            f'<td>{diag.get("skewness_status", "-")}</td></tr>\n'
            f'<tr><td>Kurtosis</td><td>{self._fmt(diag.get("kurtosis", 0), 3)}</td>'
            f'<td>{diag.get("kurtosis_status", "-")}</td></tr>\n'
            f'<tr><td>Durbin-Watson</td><td>{self._fmt(diag.get("durbin_watson", 0), 3)}</td>'
            f'<td>{diag.get("dw_status", "-")}</td></tr>\n'
            f'<tr><td>Breusch-Pagan p</td><td>{self._fmt(diag.get("breusch_pagan_p", 0), 4)}</td>'
            f'<td>{diag.get("heteroscedasticity_status", "-")}</td></tr>\n'
            '</table>'
        )

    def _html_change_point_analysis(self, data: Dict[str, Any]) -> str:
        """Render HTML change-point analysis."""
        cp = data.get("change_point_analysis", {})
        segments = cp.get("segments", [])
        rows = ""
        for seg in segments:
            rows += (
                f'<tr><td>{seg.get("name", "-")}</td>'
                f'<td>{seg.get("range", "-")}</td>'
                f'<td>{self._fmt(seg.get("slope", 0), 4)}</td>'
                f'<td>{self._fmt(seg.get("intercept", 0), 2)}</td>'
                f'<td>{self._fmt(seg.get("r_squared", 0), 4)}</td></tr>\n'
            )
        return (
            '<h2>7. Change-Point Analysis</h2>\n'
            f'<p>Model: {cp.get("model_type", "-")} | '
            f'Balance Point: {self._fmt(cp.get("balance_point", 0), 1)} deg</p>\n'
            '<table>\n<tr><th>Segment</th><th>Range</th><th>Slope</th>'
            f'<th>Intercept</th><th>R-sq</th></tr>\n{rows}</table>'
        )

    def _html_weather_normalization(self, data: Dict[str, Any]) -> str:
        """Render HTML weather normalization."""
        w = data.get("weather_normalization", {})
        return (
            '<h2>8. Weather Normalization</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">HDD Base</span>'
            f'<span class="value">{self._fmt(w.get("hdd_base_c", 0), 1)} C</span></div>\n'
            f'  <div class="card"><span class="label">CDD Base</span>'
            f'<span class="value">{self._fmt(w.get("cdd_base_c", 0), 1)} C</span></div>\n'
            f'  <div class="card"><span class="label">Baseline HDD</span>'
            f'<span class="value">{self._fmt(w.get("baseline_hdd", 0), 0)}</span></div>\n'
            f'  <div class="card"><span class="label">Baseline CDD</span>'
            f'<span class="value">{self._fmt(w.get("baseline_cdd", 0), 0)}</span></div>\n'
            f'  <div class="card"><span class="label">Normalized</span>'
            f'<span class="value">{self._fmt(w.get("normalized_mwh", 0), 1)} MWh</span></div>\n'
            '</div>'
        )

    def _html_baseline_adjustments(self, data: Dict[str, Any]) -> str:
        """Render HTML baseline adjustments table."""
        adjustments = data.get("baseline_adjustments", [])
        rows = ""
        for adj in adjustments:
            rows += (
                f'<tr><td>{adj.get("name", "-")}</td>'
                f'<td>{adj.get("type", "-")}</td>'
                f'<td>{self._fmt(adj.get("impact_mwh", 0), 1)}</td>'
                f'<td>{adj.get("justification", "-")}</td></tr>\n'
            )
        return (
            '<h2>9. Baseline Adjustments</h2>\n'
            '<table>\n<tr><th>Adjustment</th><th>Type</th>'
            f'<th>Impact (MWh)</th><th>Justification</th></tr>\n{rows}</table>'
        )

    def _html_recommendations(self, data: Dict[str, Any]) -> str:
        """Render HTML recommendations."""
        recs = data.get("recommendations", [
            "Verify CV(RMSE) remains below 25% for monthly data per ASHRAE 14",
            "Monitor residual patterns for signs of model degradation",
        ])
        items = "".join(f'<li>{r}</li>\n' for r in recs)
        return f'<h2>10. Recommendations</h2>\n<ol>\n{items}</ol>'

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_baseline_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON baseline summary."""
        s = data.get("baseline_summary", {})
        return {
            "period": s.get("period", ""),
            "duration_months": s.get("duration_months", 0),
            "data_points": s.get("data_points", 0),
            "total_consumption_mwh": s.get("total_consumption_mwh", 0),
            "avg_consumption_mwh": s.get("avg_consumption_mwh", 0),
            "data_completeness_pct": s.get("data_completeness_pct", 0),
        }

    def _json_charts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chart data payloads for visualization."""
        variables = data.get("independent_variables", [])
        candidates = data.get("model_selection", {}).get("candidates", [])
        segments = data.get("change_point_analysis", {}).get("segments", [])
        return {
            "variable_correlation": {
                "type": "bar",
                "labels": [v.get("variable", "") for v in variables],
                "values": [v.get("correlation", 0) for v in variables],
            },
            "model_comparison": {
                "type": "grouped_bar",
                "labels": [c.get("model", "") for c in candidates],
                "series": {
                    "r_squared": [c.get("r_squared", 0) for c in candidates],
                    "cvrmse": [c.get("cvrmse_pct", 0) for c in candidates],
                },
            },
            "change_point_segments": {
                "type": "multi_line",
                "segments": [
                    {
                        "name": s.get("name", ""),
                        "slope": s.get("slope", 0),
                        "intercept": s.get("intercept", 0),
                    }
                    for s in segments
                ],
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
            "h3{color:#495057;margin-top:20px;}"
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
