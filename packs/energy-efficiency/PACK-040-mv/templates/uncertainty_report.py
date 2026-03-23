# -*- coding: utf-8 -*-
"""
UncertaintyReportTemplate - Uncertainty Analysis Report for PACK-040.

Generates comprehensive uncertainty analysis reports covering measurement
uncertainty, model uncertainty, sampling uncertainty, combined fractional
savings uncertainty (FSU), minimum detectable savings calculation, and
confidence interval analysis per ASHRAE Guideline 14.

Sections:
    1. Uncertainty Summary
    2. Measurement Uncertainty
    3. Model Uncertainty
    4. Sampling Uncertainty
    5. Combined FSU
    6. Minimum Detectable Savings
    7. Sensitivity Analysis
    8. Error Propagation
    9. Confidence Intervals
    10. Recommendations

Output Formats:
    - Markdown (with provenance hash comment)
    - HTML (self-contained with inline CSS)
    - JSON (structured with chart data)

Regulatory References:
    - ASHRAE Guideline 14-2014 (uncertainty quantification)
    - IPMVP Core Concepts 2022 (FSU methodology)
    - ISO 50015:2014 (measurement uncertainty)
    - GUM (Guide to Expression of Uncertainty in Measurement)

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


class UncertaintyReportTemplate:
    """
    Uncertainty analysis report template.

    Renders comprehensive uncertainty analysis reports showing measurement,
    model, and sampling uncertainty components, combined fractional savings
    uncertainty (FSU), minimum detectable savings, sensitivity analysis,
    error propagation details, and confidence interval calculations across
    markdown, HTML, and JSON formats. All outputs include SHA-256
    provenance hashing for audit trail integrity.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize UncertaintyReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render uncertainty analysis report as Markdown.

        Args:
            data: Uncertainty engine result data.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_uncertainty_summary(data),
            self._md_measurement_uncertainty(data),
            self._md_model_uncertainty(data),
            self._md_sampling_uncertainty(data),
            self._md_combined_fsu(data),
            self._md_minimum_detectable(data),
            self._md_sensitivity_analysis(data),
            self._md_error_propagation(data),
            self._md_confidence_intervals(data),
            self._md_recommendations(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._generate_provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render uncertainty analysis report as self-contained HTML.

        Args:
            data: Uncertainty engine result data.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_uncertainty_summary(data),
            self._html_measurement_uncertainty(data),
            self._html_model_uncertainty(data),
            self._html_sampling_uncertainty(data),
            self._html_combined_fsu(data),
            self._html_minimum_detectable(data),
            self._html_sensitivity_analysis(data),
            self._html_error_propagation(data),
            self._html_confidence_intervals(data),
            self._html_recommendations(data),
        ])
        prov = self._generate_provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Uncertainty Analysis Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render uncertainty analysis report as structured JSON.

        Args:
            data: Uncertainty engine result data.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "uncertainty_report",
            "version": "40.0.0",
            "generated_at": self.generated_at.isoformat(),
            "uncertainty_summary": self._json_uncertainty_summary(data),
            "measurement_uncertainty": data.get("measurement_uncertainty", {}),
            "model_uncertainty": data.get("model_uncertainty", {}),
            "sampling_uncertainty": data.get("sampling_uncertainty", {}),
            "combined_fsu": data.get("combined_fsu", {}),
            "minimum_detectable_savings": data.get("minimum_detectable_savings", {}),
            "sensitivity_analysis": data.get("sensitivity_analysis", []),
            "error_propagation": data.get("error_propagation", {}),
            "confidence_intervals": data.get("confidence_intervals", []),
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
            f"# Uncertainty Analysis Report\n\n"
            f"**Facility:** {facility}  \n"
            f"**Reporting Period:** {data.get('reporting_period', '-')}  \n"
            f"**IPMVP Option:** {data.get('ipmvp_option', '-')}  \n"
            f"**Confidence Level:** {data.get('confidence_level', '90')}%  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-040 UncertaintyReportTemplate v40.0.0\n\n---"
        )

    def _md_uncertainty_summary(self, data: Dict[str, Any]) -> str:
        """Render uncertainty summary section."""
        s = data.get("uncertainty_summary", {})
        return (
            "## 1. Uncertainty Summary\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| Verified Savings | {self._format_energy(s.get('verified_savings_mwh', 0))} |\n"
            f"| Total Uncertainty (+/-) | {self._format_energy(s.get('total_uncertainty_mwh', 0))} |\n"
            f"| FSU | {self._fmt(s.get('fsu_pct', 0))}% |\n"
            f"| Relative Precision | {self._fmt(s.get('relative_precision_pct', 0))}% |\n"
            f"| Confidence Level | {self._fmt(s.get('confidence_level_pct', 90))}% |\n"
            f"| t-statistic | {self._fmt(s.get('t_statistic', 0), 3)} |\n"
            f"| Min Detectable Savings | {self._format_energy(s.get('min_detectable_savings_mwh', 0))} |\n"
            f"| Savings > MDS | {s.get('savings_exceeds_mds', '-')} |\n"
            f"| Overall Result | {s.get('overall_result', '-')} |"
        )

    def _md_measurement_uncertainty(self, data: Dict[str, Any]) -> str:
        """Render measurement uncertainty section."""
        meas = data.get("measurement_uncertainty", {})
        if not meas:
            return "## 2. Measurement Uncertainty\n\n_No measurement uncertainty data available._"
        meters = meas.get("meters", [])
        lines = [
            "## 2. Measurement Uncertainty\n",
            f"**Combined Measurement Uncertainty:** {self._fmt(meas.get('combined_pct', 0))}%  \n"
            f"**Method:** {meas.get('method', 'RSS')}  \n",
        ]
        if meters:
            lines.append("| Meter | Type | Accuracy (%) | Bias (%) | Random (%) | Combined (%) |")
            lines.append("|-------|------|----------:|-------:|--------:|-----------:|")
            for m in meters:
                lines.append(
                    f"| {m.get('meter_id', '-')} "
                    f"| {m.get('type', '-')} "
                    f"| {self._fmt(m.get('accuracy_pct', 0), 2)} "
                    f"| {self._fmt(m.get('bias_pct', 0), 2)} "
                    f"| {self._fmt(m.get('random_pct', 0), 2)} "
                    f"| {self._fmt(m.get('combined_pct', 0), 2)} |"
                )
        return "\n".join(lines)

    def _md_model_uncertainty(self, data: Dict[str, Any]) -> str:
        """Render model uncertainty section."""
        model = data.get("model_uncertainty", {})
        if not model:
            return "## 3. Model Uncertainty\n\n_No model uncertainty data available._"
        return (
            "## 3. Model Uncertainty\n\n"
            "| Parameter | Value |\n|-----------|-------|\n"
            f"| Model Type | {model.get('model_type', '-')} |\n"
            f"| CVRMSE | {self._fmt(model.get('cvrmse_pct', 0), 1)}% |\n"
            f"| RMSE | {self._fmt(model.get('rmse', 0), 3)} |\n"
            f"| Std Error of Estimate | {self._fmt(model.get('std_error_estimate', 0), 3)} |\n"
            f"| Degrees of Freedom | {model.get('degrees_of_freedom', '-')} |\n"
            f"| t-value (90%) | {self._fmt(model.get('t_value_90', 0), 3)} |\n"
            f"| Model Uncertainty | {self._fmt(model.get('model_uncertainty_pct', 0))}% |\n"
            f"| Prediction Interval Width | {self._format_energy(model.get('prediction_interval_mwh', 0))} |"
        )

    def _md_sampling_uncertainty(self, data: Dict[str, Any]) -> str:
        """Render sampling uncertainty section."""
        sampling = data.get("sampling_uncertainty", {})
        if not sampling:
            return "## 4. Sampling Uncertainty\n\n_No sampling uncertainty data available._"
        return (
            "## 4. Sampling Uncertainty\n\n"
            "| Parameter | Value |\n|-----------|-------|\n"
            f"| Population Size | {sampling.get('population_size', '-')} |\n"
            f"| Sample Size | {sampling.get('sample_size', '-')} |\n"
            f"| Sampling Method | {sampling.get('sampling_method', '-')} |\n"
            f"| CV of Population | {self._fmt(sampling.get('cv_population', 0))}% |\n"
            f"| Confidence Level | {self._fmt(sampling.get('confidence_level_pct', 90))}% |\n"
            f"| Precision at Confidence | {self._fmt(sampling.get('precision_pct', 0))}% |\n"
            f"| Sampling Uncertainty | {self._fmt(sampling.get('sampling_uncertainty_pct', 0))}% |\n"
            f"| Finite Population Correction | {self._fmt(sampling.get('fpc', 0), 4)} |"
        )

    def _md_combined_fsu(self, data: Dict[str, Any]) -> str:
        """Render combined FSU section."""
        fsu = data.get("combined_fsu", {})
        if not fsu:
            return "## 5. Combined Fractional Savings Uncertainty\n\n_No FSU data available._"
        components = fsu.get("components", [])
        lines = [
            "## 5. Combined Fractional Savings Uncertainty (FSU)\n",
            f"**Combination Method:** {fsu.get('method', 'RSS')}  \n"
            f"**Combined FSU:** {self._fmt(fsu.get('combined_fsu_pct', 0))}%  \n"
            f"**At Confidence:** {self._fmt(fsu.get('confidence_level_pct', 90))}%  \n"
            f"**Absolute Uncertainty:** {self._format_energy(fsu.get('absolute_uncertainty_mwh', 0))}  \n",
        ]
        if components:
            lines.append("### Uncertainty Components\n")
            lines.append("| Component | Value (%) | Contribution (%) |")
            lines.append("|-----------|--------:|---------------:|")
            for comp in components:
                lines.append(
                    f"| {comp.get('name', '-')} "
                    f"| {self._fmt(comp.get('value_pct', 0), 2)} "
                    f"| {self._fmt(comp.get('contribution_pct', 0), 1)} |"
                )
        return "\n".join(lines)

    def _md_minimum_detectable(self, data: Dict[str, Any]) -> str:
        """Render minimum detectable savings section."""
        mds = data.get("minimum_detectable_savings", {})
        if not mds:
            return "## 6. Minimum Detectable Savings\n\n_No MDS data available._"
        return (
            "## 6. Minimum Detectable Savings (MDS)\n\n"
            "| Parameter | Value |\n|-----------|-------|\n"
            f"| MDS Energy | {self._format_energy(mds.get('mds_energy_mwh', 0))} |\n"
            f"| MDS Percentage | {self._fmt(mds.get('mds_pct', 0))}% of baseline |\n"
            f"| Verified Savings | {self._format_energy(mds.get('verified_savings_mwh', 0))} |\n"
            f"| Savings / MDS Ratio | {self._fmt(mds.get('savings_mds_ratio', 0), 2)} |\n"
            f"| Savings > MDS | {mds.get('savings_exceeds_mds', '-')} |\n"
            f"| ASHRAE 14 Threshold | {self._fmt(mds.get('ashrae_threshold_pct', 50))}% at 90% |\n"
            f"| Meets ASHRAE 14 | {mds.get('meets_ashrae', '-')} |"
        )

    def _md_sensitivity_analysis(self, data: Dict[str, Any]) -> str:
        """Render sensitivity analysis section."""
        sensitivity = data.get("sensitivity_analysis", [])
        if not sensitivity:
            return "## 7. Sensitivity Analysis\n\n_No sensitivity analysis data available._"
        lines = [
            "## 7. Sensitivity Analysis\n",
            "| Parameter | Base Value | Low | High | FSU Low (%) | FSU High (%) | Impact |",
            "|-----------|--------:|----:|-----:|-----------:|------------:|--------|",
        ]
        for s in sensitivity:
            lines.append(
                f"| {s.get('parameter', '-')} "
                f"| {self._fmt(s.get('base_value', 0), 2)} "
                f"| {self._fmt(s.get('low_value', 0), 2)} "
                f"| {self._fmt(s.get('high_value', 0), 2)} "
                f"| {self._fmt(s.get('fsu_low_pct', 0), 1)} "
                f"| {self._fmt(s.get('fsu_high_pct', 0), 1)} "
                f"| {s.get('impact', '-')} |"
            )
        return "\n".join(lines)

    def _md_error_propagation(self, data: Dict[str, Any]) -> str:
        """Render error propagation section."""
        ep = data.get("error_propagation", {})
        if not ep:
            return "## 8. Error Propagation\n\n_No error propagation data available._"
        steps = ep.get("steps", [])
        lines = [
            "## 8. Error Propagation\n",
            f"**Method:** {ep.get('method', 'GUM')}  \n"
            f"**Coverage Factor (k):** {self._fmt(ep.get('coverage_factor', 0), 2)}  \n",
        ]
        if steps:
            lines.append("### Propagation Steps\n")
            lines.append("| Step | Input Uncertainty | Sensitivity Coeff | Contribution |")
            lines.append("|------|----------------:|------------------:|------------:|")
            for step in steps:
                lines.append(
                    f"| {step.get('name', '-')} "
                    f"| {self._fmt(step.get('input_uncertainty', 0), 3)} "
                    f"| {self._fmt(step.get('sensitivity_coefficient', 0), 4)} "
                    f"| {self._fmt(step.get('contribution', 0), 3)} |"
                )
        return "\n".join(lines)

    def _md_confidence_intervals(self, data: Dict[str, Any]) -> str:
        """Render confidence intervals section."""
        intervals = data.get("confidence_intervals", [])
        if not intervals:
            return "## 9. Confidence Intervals\n\n_No confidence interval data available._"
        lines = [
            "## 9. Confidence Intervals\n",
            "| Confidence Level | Lower Bound (MWh) | Savings (MWh) | Upper Bound (MWh) | Margin |",
            "|:----------------:|------------------:|-------------:|-----------------:|-------:|",
        ]
        for ci in intervals:
            lines.append(
                f"| {self._fmt(ci.get('confidence_pct', 0))}% "
                f"| {self._fmt(ci.get('lower_bound_mwh', 0), 1)} "
                f"| {self._fmt(ci.get('savings_mwh', 0), 1)} "
                f"| {self._fmt(ci.get('upper_bound_mwh', 0), 1)} "
                f"| +/- {self._fmt(ci.get('margin_mwh', 0), 1)} |"
            )
        return "\n".join(lines)

    def _md_recommendations(self, data: Dict[str, Any]) -> str:
        """Render recommendations section."""
        recs = data.get("recommendations", [])
        if not recs:
            recs = [
                "Reduce measurement uncertainty by improving meter accuracy class",
                "Increase sample size if sampling uncertainty dominates FSU",
                "Consider longer reporting period to reduce model uncertainty",
                "Target FSU below 50% at 90% confidence per ASHRAE 14",
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
            f'<h1>Uncertainty Analysis Report</h1>\n'
            f'<p class="subtitle">Facility: {facility} | '
            f'Reporting: {data.get("reporting_period", "-")} | '
            f'Confidence: {data.get("confidence_level", "90")}%</p>'
        )

    def _html_uncertainty_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML uncertainty summary cards."""
        s = data.get("uncertainty_summary", {})
        result_cls = "severity-low" if s.get("overall_result") == "PASS" else "severity-high"
        return (
            '<h2>1. Uncertainty Summary</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Savings</span>'
            f'<span class="value">{self._fmt(s.get("verified_savings_mwh", 0), 1)} MWh</span></div>\n'
            f'  <div class="card"><span class="label">FSU</span>'
            f'<span class="value">{self._fmt(s.get("fsu_pct", 0))}%</span></div>\n'
            f'  <div class="card"><span class="label">Precision</span>'
            f'<span class="value">{self._fmt(s.get("relative_precision_pct", 0))}%</span></div>\n'
            f'  <div class="card"><span class="label">MDS</span>'
            f'<span class="value">{self._fmt(s.get("min_detectable_savings_mwh", 0), 1)} MWh</span></div>\n'
            f'  <div class="card"><span class="label">Result</span>'
            f'<span class="value {result_cls}">{s.get("overall_result", "-")}</span></div>\n'
            '</div>'
        )

    def _html_measurement_uncertainty(self, data: Dict[str, Any]) -> str:
        """Render HTML measurement uncertainty table."""
        meas = data.get("measurement_uncertainty", {})
        meters = meas.get("meters", [])
        rows = ""
        for m in meters:
            rows += (
                f'<tr><td>{m.get("meter_id", "-")}</td>'
                f'<td>{m.get("type", "-")}</td>'
                f'<td>{self._fmt(m.get("accuracy_pct", 0), 2)}%</td>'
                f'<td>{self._fmt(m.get("bias_pct", 0), 2)}%</td>'
                f'<td>{self._fmt(m.get("random_pct", 0), 2)}%</td>'
                f'<td>{self._fmt(m.get("combined_pct", 0), 2)}%</td></tr>\n'
            )
        return (
            '<h2>2. Measurement Uncertainty</h2>\n'
            f'<p>Combined: {self._fmt(meas.get("combined_pct", 0))}% | '
            f'Method: {meas.get("method", "RSS")}</p>\n'
            '<table>\n<tr><th>Meter</th><th>Type</th><th>Accuracy</th>'
            f'<th>Bias</th><th>Random</th><th>Combined</th></tr>\n{rows}</table>'
        )

    def _html_model_uncertainty(self, data: Dict[str, Any]) -> str:
        """Render HTML model uncertainty cards."""
        model = data.get("model_uncertainty", {})
        return (
            '<h2>3. Model Uncertainty</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">CVRMSE</span>'
            f'<span class="value">{self._fmt(model.get("cvrmse_pct", 0), 1)}%</span></div>\n'
            f'  <div class="card"><span class="label">RMSE</span>'
            f'<span class="value">{self._fmt(model.get("rmse", 0), 3)}</span></div>\n'
            f'  <div class="card"><span class="label">Std Error</span>'
            f'<span class="value">{self._fmt(model.get("std_error_estimate", 0), 3)}</span></div>\n'
            f'  <div class="card"><span class="label">Model Unc.</span>'
            f'<span class="value">{self._fmt(model.get("model_uncertainty_pct", 0))}%</span></div>\n'
            '</div>'
        )

    def _html_sampling_uncertainty(self, data: Dict[str, Any]) -> str:
        """Render HTML sampling uncertainty table."""
        sampling = data.get("sampling_uncertainty", {})
        return (
            '<h2>4. Sampling Uncertainty</h2>\n'
            '<table>\n'
            f'<tr><th>Parameter</th><th>Value</th></tr>\n'
            f'<tr><td>Population Size</td><td>{sampling.get("population_size", "-")}</td></tr>\n'
            f'<tr><td>Sample Size</td><td>{sampling.get("sample_size", "-")}</td></tr>\n'
            f'<tr><td>Sampling Method</td><td>{sampling.get("sampling_method", "-")}</td></tr>\n'
            f'<tr><td>CV of Population</td><td>{self._fmt(sampling.get("cv_population", 0))}%</td></tr>\n'
            f'<tr><td>Precision</td><td>{self._fmt(sampling.get("precision_pct", 0))}%</td></tr>\n'
            f'<tr><td>Sampling Uncertainty</td><td>{self._fmt(sampling.get("sampling_uncertainty_pct", 0))}%</td></tr>\n'
            '</table>'
        )

    def _html_combined_fsu(self, data: Dict[str, Any]) -> str:
        """Render HTML combined FSU section."""
        fsu = data.get("combined_fsu", {})
        components = fsu.get("components", [])
        rows = ""
        for comp in components:
            rows += (
                f'<tr><td>{comp.get("name", "-")}</td>'
                f'<td>{self._fmt(comp.get("value_pct", 0), 2)}%</td>'
                f'<td>{self._fmt(comp.get("contribution_pct", 0), 1)}%</td></tr>\n'
            )
        return (
            '<h2>5. Combined FSU</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Combined FSU</span>'
            f'<span class="value">{self._fmt(fsu.get("combined_fsu_pct", 0))}%</span></div>\n'
            f'  <div class="card"><span class="label">Absolute</span>'
            f'<span class="value">{self._fmt(fsu.get("absolute_uncertainty_mwh", 0), 1)} MWh</span></div>\n'
            '</div>\n'
            '<table>\n<tr><th>Component</th><th>Value (%)</th>'
            f'<th>Contribution (%)</th></tr>\n{rows}</table>'
        )

    def _html_minimum_detectable(self, data: Dict[str, Any]) -> str:
        """Render HTML minimum detectable savings."""
        mds = data.get("minimum_detectable_savings", {})
        meets_cls = "severity-low" if mds.get("meets_ashrae") else "severity-high"
        return (
            '<h2>6. Minimum Detectable Savings</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">MDS Energy</span>'
            f'<span class="value">{self._fmt(mds.get("mds_energy_mwh", 0), 1)} MWh</span></div>\n'
            f'  <div class="card"><span class="label">MDS %</span>'
            f'<span class="value">{self._fmt(mds.get("mds_pct", 0))}%</span></div>\n'
            f'  <div class="card"><span class="label">Savings/MDS</span>'
            f'<span class="value">{self._fmt(mds.get("savings_mds_ratio", 0), 2)}</span></div>\n'
            f'  <div class="card"><span class="label">ASHRAE 14</span>'
            f'<span class="value {meets_cls}">{mds.get("meets_ashrae", "-")}</span></div>\n'
            '</div>'
        )

    def _html_sensitivity_analysis(self, data: Dict[str, Any]) -> str:
        """Render HTML sensitivity analysis table."""
        sensitivity = data.get("sensitivity_analysis", [])
        rows = ""
        for s in sensitivity:
            rows += (
                f'<tr><td>{s.get("parameter", "-")}</td>'
                f'<td>{self._fmt(s.get("base_value", 0), 2)}</td>'
                f'<td>{self._fmt(s.get("low_value", 0), 2)}</td>'
                f'<td>{self._fmt(s.get("high_value", 0), 2)}</td>'
                f'<td>{self._fmt(s.get("fsu_low_pct", 0), 1)}%</td>'
                f'<td>{self._fmt(s.get("fsu_high_pct", 0), 1)}%</td>'
                f'<td>{s.get("impact", "-")}</td></tr>\n'
            )
        return (
            '<h2>7. Sensitivity Analysis</h2>\n'
            '<table>\n<tr><th>Parameter</th><th>Base</th><th>Low</th>'
            f'<th>High</th><th>FSU Low</th><th>FSU High</th><th>Impact</th></tr>\n{rows}</table>'
        )

    def _html_error_propagation(self, data: Dict[str, Any]) -> str:
        """Render HTML error propagation table."""
        ep = data.get("error_propagation", {})
        steps = ep.get("steps", [])
        rows = ""
        for step in steps:
            rows += (
                f'<tr><td>{step.get("name", "-")}</td>'
                f'<td>{self._fmt(step.get("input_uncertainty", 0), 3)}</td>'
                f'<td>{self._fmt(step.get("sensitivity_coefficient", 0), 4)}</td>'
                f'<td>{self._fmt(step.get("contribution", 0), 3)}</td></tr>\n'
            )
        return (
            '<h2>8. Error Propagation</h2>\n'
            f'<p>Method: {ep.get("method", "GUM")} | '
            f'Coverage Factor: {self._fmt(ep.get("coverage_factor", 0), 2)}</p>\n'
            '<table>\n<tr><th>Step</th><th>Input Unc.</th>'
            f'<th>Sensitivity</th><th>Contribution</th></tr>\n{rows}</table>'
        )

    def _html_confidence_intervals(self, data: Dict[str, Any]) -> str:
        """Render HTML confidence intervals table."""
        intervals = data.get("confidence_intervals", [])
        rows = ""
        for ci in intervals:
            rows += (
                f'<tr><td>{self._fmt(ci.get("confidence_pct", 0))}%</td>'
                f'<td>{self._fmt(ci.get("lower_bound_mwh", 0), 1)}</td>'
                f'<td>{self._fmt(ci.get("savings_mwh", 0), 1)}</td>'
                f'<td>{self._fmt(ci.get("upper_bound_mwh", 0), 1)}</td>'
                f'<td>+/- {self._fmt(ci.get("margin_mwh", 0), 1)}</td></tr>\n'
            )
        return (
            '<h2>9. Confidence Intervals</h2>\n'
            '<table>\n<tr><th>Confidence</th><th>Lower (MWh)</th><th>Savings (MWh)</th>'
            f'<th>Upper (MWh)</th><th>Margin</th></tr>\n{rows}</table>'
        )

    def _html_recommendations(self, data: Dict[str, Any]) -> str:
        """Render HTML recommendations."""
        recs = data.get("recommendations", [
            "Reduce measurement uncertainty by improving meter accuracy class",
            "Target FSU below 50% at 90% confidence per ASHRAE 14",
        ])
        items = "".join(f'<li>{r}</li>\n' for r in recs)
        return f'<h2>10. Recommendations</h2>\n<ol>\n{items}</ol>'

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_uncertainty_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON uncertainty summary."""
        s = data.get("uncertainty_summary", {})
        return {
            "verified_savings_mwh": s.get("verified_savings_mwh", 0),
            "total_uncertainty_mwh": s.get("total_uncertainty_mwh", 0),
            "fsu_pct": s.get("fsu_pct", 0),
            "relative_precision_pct": s.get("relative_precision_pct", 0),
            "confidence_level_pct": s.get("confidence_level_pct", 90),
            "min_detectable_savings_mwh": s.get("min_detectable_savings_mwh", 0),
            "overall_result": s.get("overall_result", ""),
        }

    def _json_charts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chart data payloads for visualization."""
        components = data.get("combined_fsu", {}).get("components", [])
        sensitivity = data.get("sensitivity_analysis", [])
        intervals = data.get("confidence_intervals", [])
        return {
            "uncertainty_breakdown": {
                "type": "pie",
                "labels": [c.get("name", "") for c in components],
                "values": [c.get("contribution_pct", 0) for c in components],
            },
            "sensitivity_tornado": {
                "type": "tornado",
                "labels": [s.get("parameter", "") for s in sensitivity],
                "low": [s.get("fsu_low_pct", 0) for s in sensitivity],
                "high": [s.get("fsu_high_pct", 0) for s in sensitivity],
            },
            "confidence_intervals": {
                "type": "error_bar",
                "labels": [f'{ci.get("confidence_pct", 0)}%' for ci in intervals],
                "values": [ci.get("savings_mwh", 0) for ci in intervals],
                "lower": [ci.get("lower_bound_mwh", 0) for ci in intervals],
                "upper": [ci.get("upper_bound_mwh", 0) for ci in intervals],
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
