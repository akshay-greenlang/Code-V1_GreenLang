# -*- coding: utf-8 -*-
"""
EnergyBaselineReportTemplate - Energy baseline establishment report for PACK-031.

Generates baseline establishment reports with regression model results,
Energy Performance Indicator (EnPI) charts data, CUSUM analysis, statistical
validation, degree-day normalization, and energy balance verification.
Follows ISO 50006:2014 and ISO 50015:2014 methodology.

Sections:
    1. Executive Summary
    2. Baseline Period Definition
    3. Relevant Variables & Data Collection
    4. Regression Model Results
    5. EnPI Definition & Charts
    6. CUSUM Analysis
    7. Degree-Day Normalization
    8. Energy Balance
    9. Statistical Validation

Author: GreenLang Team
Version: 31.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class EnergyBaselineReportTemplate:
    """
    Energy baseline establishment report template.

    Renders statistical baseline models, EnPI tracking, CUSUM charts,
    and degree-day normalization results across markdown, HTML, and JSON.
    Compliant with ISO 50006 and ISO 50015 methodology.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    STATISTICAL_THRESHOLDS: Dict[str, float] = {
        "r_squared_min": 0.75,
        "cv_rmse_max_pct": 25.0,
        "t_stat_min": 2.0,
        "p_value_max": 0.05,
        "f_stat_significance": 0.05,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize EnergyBaselineReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render energy baseline report as Markdown.

        Args:
            data: Baseline engine result data.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_executive_summary(data),
            self._md_baseline_period(data),
            self._md_relevant_variables(data),
            self._md_regression_model(data),
            self._md_enpi_definition(data),
            self._md_cusum_analysis(data),
            self._md_degree_day_normalization(data),
            self._md_energy_balance(data),
            self._md_statistical_validation(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render energy baseline report as self-contained HTML.

        Args:
            data: Baseline engine result data.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_executive_summary(data),
            self._html_regression_model(data),
            self._html_enpi_charts(data),
            self._html_cusum_chart(data),
            self._html_degree_day(data),
            self._html_energy_balance(data),
            self._html_statistical_validation(data),
        ])
        prov = self._provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Energy Baseline Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render energy baseline report as structured JSON.

        Args:
            data: Baseline engine result data.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "energy_baseline_report",
            "version": "31.0.0",
            "standards": ["ISO 50006:2014", "ISO 50015:2014"],
            "generated_at": self.generated_at.isoformat(),
            "executive_summary": self._json_executive_summary(data),
            "baseline_period": data.get("baseline_period", {}),
            "relevant_variables": data.get("relevant_variables", []),
            "regression_model": data.get("regression_model", {}),
            "enpi_definition": data.get("enpi_definition", {}),
            "cusum_analysis": data.get("cusum_analysis", {}),
            "degree_day_normalization": data.get("degree_day_normalization", {}),
            "energy_balance": data.get("energy_balance", {}),
            "statistical_validation": self._json_statistical_validation(data),
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
        return (
            f"# Energy Baseline Report\n\n"
            f"**Facility:** {facility}  \n"
            f"**Standards:** ISO 50006:2014 / ISO 50015:2014  \n"
            f"**Generated:** {ts}  \n"
            f"**Template:** PACK-031 EnergyBaselineReportTemplate v31.0.0\n\n---"
        )

    def _md_executive_summary(self, data: Dict[str, Any]) -> str:
        """Render executive summary."""
        summary = data.get("executive_summary", {})
        model = data.get("regression_model", {})
        return (
            "## 1. Executive Summary\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| Baseline Period | {summary.get('baseline_period', '-')} |\n"
            f"| Total Baseline Consumption | {self._fmt(summary.get('total_baseline_mwh', 0))} MWh |\n"
            f"| Model Type | {model.get('model_type', 'Multivariable Linear Regression')} |\n"
            f"| R-Squared | {self._fmt(model.get('r_squared', 0), 4)} |\n"
            f"| CV(RMSE) | {self._fmt(model.get('cv_rmse_pct', 0), 1)}% |\n"
            f"| Baseline EnPI | {self._fmt(summary.get('baseline_enpi', 0))} "
            f"{summary.get('enpi_unit', 'kWh/unit')} |\n"
            f"| Model Validity | {self._validity_badge(model)} |"
        )

    def _md_baseline_period(self, data: Dict[str, Any]) -> str:
        """Render baseline period definition."""
        period = data.get("baseline_period", {})
        return (
            "## 2. Baseline Period Definition\n\n"
            f"- **Start Date:** {period.get('start_date', '-')}\n"
            f"- **End Date:** {period.get('end_date', '-')}\n"
            f"- **Duration:** {period.get('duration_months', 12)} months\n"
            f"- **Data Points:** {period.get('data_points', 0)}\n"
            f"- **Data Interval:** {period.get('interval', 'Monthly')}\n"
            f"- **Exclusions:** {period.get('exclusions', 'None')}\n"
            f"- **Justification:** {period.get('justification', 'Representative operating conditions')}"
        )

    def _md_relevant_variables(self, data: Dict[str, Any]) -> str:
        """Render relevant variables and data collection."""
        variables = data.get("relevant_variables", [])
        if not variables:
            return "## 3. Relevant Variables\n\n_No relevant variables defined._"
        lines = [
            "## 3. Relevant Variables & Data Collection\n",
            "| Variable | Type | Source | Unit | Correlation | Significance |",
            "|----------|------|--------|------|-------------|-------------|",
        ]
        for v in variables:
            lines.append(
                f"| {v.get('name', '-')} "
                f"| {v.get('type', '-')} "
                f"| {v.get('source', '-')} "
                f"| {v.get('unit', '-')} "
                f"| {self._fmt(v.get('correlation', 0), 3)} "
                f"| {v.get('significance', '-')} |"
            )
        return "\n".join(lines)

    def _md_regression_model(self, data: Dict[str, Any]) -> str:
        """Render regression model results."""
        model = data.get("regression_model", {})
        coefficients = model.get("coefficients", [])
        lines = [
            "## 4. Regression Model Results\n",
            f"**Model Type:** {model.get('model_type', 'Multivariable Linear Regression')}  ",
            f"**Dependent Variable:** {model.get('dependent_variable', 'Energy Consumption (kWh)')}  ",
            f"**Equation:** {model.get('equation', 'E = a + b1*X1 + b2*X2 + ...')}",
        ]
        if coefficients:
            lines.extend([
                "\n### Model Coefficients\n",
                "| Variable | Coefficient | Std Error | t-Statistic | p-Value |",
                "|----------|------------|-----------|-------------|---------|",
            ])
            for c in coefficients:
                lines.append(
                    f"| {c.get('variable', '-')} "
                    f"| {self._fmt(c.get('coefficient', 0), 4)} "
                    f"| {self._fmt(c.get('std_error', 0), 4)} "
                    f"| {self._fmt(c.get('t_statistic', 0), 3)} "
                    f"| {self._fmt(c.get('p_value', 0), 4)} |"
                )
        lines.extend([
            "\n### Model Fit Statistics\n",
            "| Statistic | Value | Threshold | Status |",
            "|-----------|-------|-----------|--------|",
            f"| R-Squared | {self._fmt(model.get('r_squared', 0), 4)} "
            f"| >= {self.STATISTICAL_THRESHOLDS['r_squared_min']} "
            f"| {self._pass_fail(model.get('r_squared', 0), self.STATISTICAL_THRESHOLDS['r_squared_min'], 'gte')} |",
            f"| CV(RMSE) | {self._fmt(model.get('cv_rmse_pct', 0), 2)}% "
            f"| <= {self.STATISTICAL_THRESHOLDS['cv_rmse_max_pct']}% "
            f"| {self._pass_fail(model.get('cv_rmse_pct', 100), self.STATISTICAL_THRESHOLDS['cv_rmse_max_pct'], 'lte')} |",
            f"| F-Statistic | {self._fmt(model.get('f_statistic', 0), 2)} "
            f"| p < {self.STATISTICAL_THRESHOLDS['f_stat_significance']} "
            f"| {self._pass_fail(model.get('f_p_value', 1), self.STATISTICAL_THRESHOLDS['f_stat_significance'], 'lte')} |",
        ])
        return "\n".join(lines)

    def _md_enpi_definition(self, data: Dict[str, Any]) -> str:
        """Render EnPI definition and charts data."""
        enpi = data.get("enpi_definition", {})
        indicators = enpi.get("indicators", [])
        lines = [
            "## 5. Energy Performance Indicators (EnPI)\n",
            f"**Baseline EnPI:** {self._fmt(enpi.get('baseline_value', 0))} "
            f"{enpi.get('unit', 'kWh/unit')}  ",
            f"**Target EnPI:** {self._fmt(enpi.get('target_value', 0))} "
            f"{enpi.get('unit', 'kWh/unit')}  ",
            f"**Improvement Target:** {self._fmt(enpi.get('improvement_target_pct', 0))}%",
        ]
        if indicators:
            lines.extend([
                "\n### EnPI Tracking\n",
                "| Period | Actual EnPI | Baseline EnPI | Variance | Status |",
                "|--------|-----------|--------------|----------|--------|",
            ])
            for ind in indicators:
                lines.append(
                    f"| {ind.get('period', '-')} "
                    f"| {self._fmt(ind.get('actual', 0))} "
                    f"| {self._fmt(ind.get('baseline', 0))} "
                    f"| {self._fmt(ind.get('variance_pct', 0))}% "
                    f"| {ind.get('status', '-')} |"
                )
        return "\n".join(lines)

    def _md_cusum_analysis(self, data: Dict[str, Any]) -> str:
        """Render CUSUM analysis section."""
        cusum = data.get("cusum_analysis", {})
        points = cusum.get("data_points", [])
        lines = [
            "## 6. CUSUM Analysis\n",
            f"**Method:** Cumulative Sum of differences between actual and predicted  ",
            f"**Trend Direction:** {cusum.get('trend_direction', '-')}  ",
            f"**Cumulative Savings:** {self._fmt(cusum.get('cumulative_savings_mwh', 0))} MWh  ",
            f"**Change Point Detected:** {cusum.get('change_point', 'None')}",
        ]
        if points:
            lines.extend([
                "\n### CUSUM Data Points\n",
                "| Period | Actual (MWh) | Predicted (MWh) | Difference | Cumulative Sum |",
                "|--------|-------------|----------------|------------|----------------|",
            ])
            for p in points[:24]:
                lines.append(
                    f"| {p.get('period', '-')} "
                    f"| {self._fmt(p.get('actual_mwh', 0))} "
                    f"| {self._fmt(p.get('predicted_mwh', 0))} "
                    f"| {self._fmt(p.get('difference_mwh', 0))} "
                    f"| {self._fmt(p.get('cumulative_sum', 0))} |"
                )
        return "\n".join(lines)

    def _md_degree_day_normalization(self, data: Dict[str, Any]) -> str:
        """Render degree-day normalization results."""
        dd = data.get("degree_day_normalization", {})
        if not dd:
            return "## 7. Degree-Day Normalization\n\n_No degree-day data available._"
        lines = [
            "## 7. Degree-Day Normalization\n",
            f"**Base Temperature (Heating):** {dd.get('hdd_base_temp_c', 15.5)} C  ",
            f"**Base Temperature (Cooling):** {dd.get('cdd_base_temp_c', 18.0)} C  ",
            f"**Weather Station:** {dd.get('weather_station', '-')}  ",
            f"**Heating Coefficient:** {self._fmt(dd.get('heating_coefficient', 0), 4)} kWh/HDD  ",
            f"**Cooling Coefficient:** {self._fmt(dd.get('cooling_coefficient', 0), 4)} kWh/CDD  ",
            f"**Baseload:** {self._fmt(dd.get('baseload_kwh', 0))} kWh/period",
        ]
        monthly = dd.get("monthly_normalized", [])
        if monthly:
            lines.extend([
                "\n### Monthly Normalized Consumption\n",
                "| Month | HDD | CDD | Actual (MWh) | Normalized (MWh) | Adjustment |",
                "|-------|-----|-----|-------------|------------------|------------|",
            ])
            for m in monthly:
                lines.append(
                    f"| {m.get('month', '-')} "
                    f"| {self._fmt(m.get('hdd', 0), 0)} "
                    f"| {self._fmt(m.get('cdd', 0), 0)} "
                    f"| {self._fmt(m.get('actual_mwh', 0))} "
                    f"| {self._fmt(m.get('normalized_mwh', 0))} "
                    f"| {self._fmt(m.get('adjustment_mwh', 0))} |"
                )
        return "\n".join(lines)

    def _md_energy_balance(self, data: Dict[str, Any]) -> str:
        """Render energy balance section."""
        balance = data.get("energy_balance", {})
        inputs = balance.get("inputs", [])
        outputs = balance.get("outputs", [])
        lines = [
            "## 8. Energy Balance\n",
            f"**Total Energy Input:** {self._fmt(balance.get('total_input_mwh', 0))} MWh  ",
            f"**Total Energy Output/Use:** {self._fmt(balance.get('total_output_mwh', 0))} MWh  ",
            f"**Unaccounted:** {self._fmt(balance.get('unaccounted_mwh', 0))} MWh "
            f"({self._fmt(balance.get('unaccounted_pct', 0))}%)  ",
            f"**Balance Status:** {balance.get('status', 'Not Verified')}",
        ]
        if inputs:
            lines.extend([
                "\n### Energy Inputs\n",
                "| Source | Energy (MWh) | Share (%) |",
                "|--------|-------------|-----------|",
            ])
            for inp in inputs:
                lines.append(
                    f"| {inp.get('source', '-')} "
                    f"| {self._fmt(inp.get('energy_mwh', 0))} "
                    f"| {self._fmt(inp.get('share_pct', 0))}% |"
                )
        if outputs:
            lines.extend([
                "\n### Energy Uses\n",
                "| End Use | Energy (MWh) | Share (%) |",
                "|---------|-------------|-----------|",
            ])
            for out in outputs:
                lines.append(
                    f"| {out.get('end_use', '-')} "
                    f"| {self._fmt(out.get('energy_mwh', 0))} "
                    f"| {self._fmt(out.get('share_pct', 0))}% |"
                )
        return "\n".join(lines)

    def _md_statistical_validation(self, data: Dict[str, Any]) -> str:
        """Render statistical validation summary."""
        model = data.get("regression_model", {})
        validation = data.get("statistical_validation", {})
        residuals = validation.get("residual_analysis", {})
        lines = [
            "## 9. Statistical Validation\n",
            "### Overall Model Validity\n",
            "| Test | Result | Criterion | Pass/Fail |",
            "|------|--------|-----------|-----------|",
            f"| R-Squared | {self._fmt(model.get('r_squared', 0), 4)} "
            f"| >= 0.75 | {self._pass_fail(model.get('r_squared', 0), 0.75, 'gte')} |",
            f"| CV(RMSE) | {self._fmt(model.get('cv_rmse_pct', 0), 2)}% "
            f"| <= 25% | {self._pass_fail(model.get('cv_rmse_pct', 100), 25.0, 'lte')} |",
            f"| Normality (Shapiro-Wilk) | p={self._fmt(residuals.get('shapiro_p_value', 0), 4)} "
            f"| p > 0.05 | {self._pass_fail(residuals.get('shapiro_p_value', 0), 0.05, 'gte')} |",
            f"| Autocorrelation (DW) | {self._fmt(residuals.get('durbin_watson', 0), 3)} "
            f"| 1.5 - 2.5 | {self._dw_status(residuals.get('durbin_watson', 0))} |",
            f"| Homoscedasticity | {residuals.get('homoscedasticity_test', '-')} "
            f"| p > 0.05 | {residuals.get('homoscedasticity_status', '-')} |",
        ]
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render report footer."""
        return "---\n\n*Generated by GreenLang PACK-031 Industrial Energy Audit Pack*"

    # ------------------------------------------------------------------
    # HTML section renderers
    # ------------------------------------------------------------------

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        facility = data.get("facility_name", "Industrial Facility")
        return (
            '<h1>Energy Baseline Report</h1>\n'
            f'<p class="subtitle">Facility: {facility} | '
            f'Standards: ISO 50006 / ISO 50015</p>'
        )

    def _html_executive_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML executive summary cards."""
        s = data.get("executive_summary", {})
        model = data.get("regression_model", {})
        return (
            '<h2>Executive Summary</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Baseline Consumption</span>'
            f'<span class="value">{self._fmt(s.get("total_baseline_mwh", 0))} MWh</span></div>\n'
            f'  <div class="card"><span class="label">R-Squared</span>'
            f'<span class="value">{self._fmt(model.get("r_squared", 0), 4)}</span></div>\n'
            f'  <div class="card"><span class="label">CV(RMSE)</span>'
            f'<span class="value">{self._fmt(model.get("cv_rmse_pct", 0), 1)}%</span></div>\n'
            f'  <div class="card"><span class="label">Baseline EnPI</span>'
            f'<span class="value">{self._fmt(s.get("baseline_enpi", 0))} '
            f'{s.get("enpi_unit", "kWh/unit")}</span></div>\n'
            '</div>'
        )

    def _html_regression_model(self, data: Dict[str, Any]) -> str:
        """Render HTML regression model section."""
        model = data.get("regression_model", {})
        return (
            '<h2>Regression Model</h2>\n'
            f'<p><strong>Equation:</strong> {model.get("equation", "-")}</p>\n'
            f'<p>R² = {self._fmt(model.get("r_squared", 0), 4)}, '
            f'CV(RMSE) = {self._fmt(model.get("cv_rmse_pct", 0), 2)}%</p>'
        )

    def _html_enpi_charts(self, data: Dict[str, Any]) -> str:
        """Render HTML EnPI charts placeholder."""
        enpi = data.get("enpi_definition", {})
        return (
            '<h2>Energy Performance Indicators</h2>\n'
            f'<p>Baseline: {self._fmt(enpi.get("baseline_value", 0))} '
            f'{enpi.get("unit", "kWh/unit")} | '
            f'Target: {self._fmt(enpi.get("target_value", 0))} '
            f'{enpi.get("unit", "kWh/unit")}</p>\n'
            '<div class="chart-placeholder" data-chart="enpi_trend">'
            '[EnPI Trend Chart]</div>'
        )

    def _html_cusum_chart(self, data: Dict[str, Any]) -> str:
        """Render HTML CUSUM chart placeholder."""
        cusum = data.get("cusum_analysis", {})
        return (
            '<h2>CUSUM Analysis</h2>\n'
            f'<p>Cumulative Savings: '
            f'{self._fmt(cusum.get("cumulative_savings_mwh", 0))} MWh</p>\n'
            '<div class="chart-placeholder" data-chart="cusum">'
            '[CUSUM Chart]</div>'
        )

    def _html_degree_day(self, data: Dict[str, Any]) -> str:
        """Render HTML degree-day normalization."""
        dd = data.get("degree_day_normalization", {})
        return (
            '<h2>Degree-Day Normalization</h2>\n'
            f'<p>HDD Base: {dd.get("hdd_base_temp_c", 15.5)} C | '
            f'CDD Base: {dd.get("cdd_base_temp_c", 18.0)} C</p>'
        )

    def _html_energy_balance(self, data: Dict[str, Any]) -> str:
        """Render HTML energy balance."""
        balance = data.get("energy_balance", {})
        return (
            '<h2>Energy Balance</h2>\n'
            f'<p>Input: {self._fmt(balance.get("total_input_mwh", 0))} MWh | '
            f'Output: {self._fmt(balance.get("total_output_mwh", 0))} MWh | '
            f'Unaccounted: {self._fmt(balance.get("unaccounted_pct", 0))}%</p>'
        )

    def _html_statistical_validation(self, data: Dict[str, Any]) -> str:
        """Render HTML statistical validation summary."""
        model = data.get("regression_model", {})
        validity = self._validity_badge(model)
        color = "#059669" if validity == "VALID" else "#dc2626"
        return (
            '<h2>Statistical Validation</h2>\n'
            f'<div class="validity-badge" style="color:{color};">'
            f'Model Status: <strong>{validity}</strong></div>'
        )

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_executive_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON executive summary."""
        s = data.get("executive_summary", {})
        model = data.get("regression_model", {})
        return {
            "total_baseline_mwh": s.get("total_baseline_mwh", 0),
            "baseline_enpi": s.get("baseline_enpi", 0),
            "enpi_unit": s.get("enpi_unit", "kWh/unit"),
            "r_squared": model.get("r_squared", 0),
            "cv_rmse_pct": model.get("cv_rmse_pct", 0),
            "model_validity": self._validity_badge(model),
        }

    def _json_statistical_validation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON statistical validation."""
        model = data.get("regression_model", {})
        validation = data.get("statistical_validation", {})
        return {
            "model_valid": self._validity_badge(model) == "VALID",
            "r_squared": model.get("r_squared", 0),
            "cv_rmse_pct": model.get("cv_rmse_pct", 0),
            "residual_analysis": validation.get("residual_analysis", {}),
            "thresholds": self.STATISTICAL_THRESHOLDS,
        }

    def _json_charts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chart data payloads for visualization."""
        cusum = data.get("cusum_analysis", {}).get("data_points", [])
        enpi = data.get("enpi_definition", {}).get("indicators", [])
        dd = data.get("degree_day_normalization", {}).get("monthly_normalized", [])
        return {
            "enpi_trend": {
                "type": "line",
                "labels": [e.get("period", "") for e in enpi],
                "series": {
                    "actual": [e.get("actual", 0) for e in enpi],
                    "baseline": [e.get("baseline", 0) for e in enpi],
                },
            },
            "cusum_chart": {
                "type": "line",
                "labels": [p.get("period", "") for p in cusum],
                "values": [p.get("cumulative_sum", 0) for p in cusum],
            },
            "regression_scatter": {
                "type": "scatter",
                "x_label": data.get("regression_model", {}).get(
                    "primary_variable", "Production"
                ),
                "y_label": "Energy Consumption",
            },
            "degree_day_bar": {
                "type": "grouped_bar",
                "labels": [m.get("month", "") for m in dd],
                "series": {
                    "actual": [m.get("actual_mwh", 0) for m in dd],
                    "normalized": [m.get("normalized_mwh", 0) for m in dd],
                },
            },
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _validity_badge(self, model: Dict[str, Any]) -> str:
        """Determine model validity status.

        Args:
            model: Regression model results dict.

        Returns:
            'VALID' or 'INVALID' string.
        """
        r2 = model.get("r_squared", 0)
        cv = model.get("cv_rmse_pct", 100)
        if r2 >= self.STATISTICAL_THRESHOLDS["r_squared_min"] and \
           cv <= self.STATISTICAL_THRESHOLDS["cv_rmse_max_pct"]:
            return "VALID"
        return "INVALID"

    def _pass_fail(self, value: float, threshold: float, op: str) -> str:
        """Evaluate pass/fail against threshold.

        Args:
            value: Measured value.
            threshold: Threshold value.
            op: Comparison operator ('gte' or 'lte').

        Returns:
            'PASS' or 'FAIL' string.
        """
        if op == "gte":
            return "PASS" if value >= threshold else "FAIL"
        elif op == "lte":
            return "PASS" if value <= threshold else "FAIL"
        return "N/A"

    def _dw_status(self, dw: float) -> str:
        """Evaluate Durbin-Watson statistic.

        Args:
            dw: Durbin-Watson value (ideal range 1.5-2.5).

        Returns:
            'PASS' or 'FAIL' string.
        """
        return "PASS" if 1.5 <= dw <= 2.5 else "FAIL"

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
            ".summary-cards{display:flex;gap:15px;margin:15px 0;}"
            ".card{background:#f8f9fa;border-radius:8px;padding:15px;flex:1;text-align:center;}"
            ".label{display:block;font-size:0.85em;color:#6c757d;}"
            ".value{display:block;font-size:1.4em;font-weight:700;color:#198754;}"
            ".chart-placeholder{background:#f0f0f0;border:2px dashed #ccc;padding:40px;text-align:center;margin:15px 0;}"
            ".validity-badge{font-size:1.2em;padding:10px;border-radius:6px;display:inline-block;}"
            ".subtitle{color:#6c757d;font-size:0.95em;}"
        )

    def _fmt(self, val: Any, decimals: int = 2) -> str:
        """Format a numeric value with comma separators."""
        if isinstance(val, (int, float)):
            return f"{val:,.{decimals}f}"
        return str(val)

    def _provenance(self, content: str) -> str:
        """Compute SHA-256 provenance hash."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
